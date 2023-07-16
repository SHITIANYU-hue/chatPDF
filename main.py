import logging

logging.basicConfig(level=logging.CRITICAL)
import re
import os
from pathlib import Path
import time
import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from llama_index import (
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext,
    StorageContext,
    download_loader,
    load_index_from_storage,
)
from utils import CACHE, FILES, models, cls, handle_save, handle_exit, initialize, select_file

load_dotenv()
os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.environ["OPENAI_API_KEY"]
history = []
NUM_ROUND=3
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.618, model_name=models["gpt-3"], max_tokens=256))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=1024)


def make_index(file):
    cls()
    print("👀 Loading...")

    PDFReader = download_loader("PDFReader")
    loader = PDFReader()
    documents = loader.load_data(file=Path(FILES) / file)

    if os.path.exists(Path(CACHE) / file):
        print("📚 Index found in cache")
        return
    else:
        print("📚 Index not found in cache, creating it...")
        index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
        index.storage_context.persist(persist_dir=Path(CACHE) / file)

def generate_chat_completion(prompt):
    try:
        response =  openai.ChatCompletion.create(
        model='gpt-3.5-turbo',messages=[{'role':'user','content':prompt}]
        )
        usage = response["usage"]["total_tokens"]
        return response.choices[0].message.content,usage
    
    except openai.error.RateLimitError as e:

        retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
        print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return generate_chat_completion(prompt)

    except openai.error.ServiceUnavailableError as e:
        retry_time = 10  # Adjust the retry time as needed
        print(f"Service is unavailable. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return generate_chat_completion(prompt)

    except openai.error.APIError as e:
        retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
        print(f"API error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return generate_chat_completion(prompt)

    except OSError as e:
        retry_time = 5  # Adjust the retry time as needed
        print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")      
        time.sleep(retry_time)
        return generate_chat_completion(prompt)   
    
def generate_interview_summary(context_history,round_sum=True):
    if round_sum:
        context=str(f'Assume you are the interviewer, based on the discussion:{context_history[-1]},provide an assessment for the response of interviewee:')
    else:
        context=str(f'Assume you are the interviewer, based on the discussion:{context_history},provide the final assessment for the response of interviewee:')
    return generate_chat_completion(context)

    
def follow_up(context_history):
    context=str(f'Assume you are the interviewer, based on the response from interviewee:{context_history}, provide a follow up for the response:')
    return generate_chat_completion(context)



def interview_assistant(index):
    resume = "Please provide a brief summary of your resume: "

    # 提取简历信息并设置对话上下文,生成问题
    # context = extract_resume_information(resume)
    context = 'briefly and logically  generate 5 interview questions for him based on his resume, from the aspect of behavior, coding skill, system design, research experience, please only provide the question:'
    context_history = []
    query_engine = index.as_query_engine(response_mode="compact")
    response = str(query_engine.query(context))
    # print('response',response)
    questions = re.findall(r"\d+\.\s(.*?)\?", response)
    # Remove the numbering from the questions
    # Print the individual questions
    # for question in questions:
    #     print(question)

    for j in range(len(questions)):
        print(f'\n round {j}❓：')
        print(questions[j])
        # print("\n👻 Response: " + str(response))

        # # 根据需要进行对话结束的判断
        # if should_end_interview(context_history):
        #     break
        for i in range(NUM_ROUND):
            prompt = input("\n😎 interviewee: ")
            if prompt == "exit":
                handle_exit()
            elif prompt == "save":
                handle_save(str(file_name), context_history)
            response=follow_up(prompt)
            print('\n interviewer：')
            print(response[0])
            # 保存对话历史记录和评价
            context_history.append({"interviewee": str(prompt), "interviewer": str(response)})

        prompt = input("\n😎 interviewee: ")
        if prompt == "exit":
            handle_exit()
        elif prompt == "save":
            handle_save(str(file_name), context_history)
        response=follow_up(prompt)
        context_history.append({"interviewee": str(prompt), "interviewer": str(response)})
        # 生成面试总结
        interview_summary = generate_interview_summary(context_history)

        # 展示面试总结
        print(f"\n📝 Interview Summary for  round {j}: ")
        print(interview_summary[0])
    interview_summary_final=generate_interview_summary(context_history, round_sum=False)
    print("\n📝 Your final assessment: ")
    print(interview_summary_final[0])

    

def ask(file_name):
    try:
        print("👀 Loading...")
        storage_context = StorageContext.from_defaults(persist_dir=Path(CACHE) / file_name)
        index = load_index_from_storage(storage_context, service_context=service_context)
        cls()
        print("✅ Ready! Let's start the interview")
        print("ℹ️ Press Ctrl+C to exit")
        interview_assistant(index)
    except KeyboardInterrupt:
        handle_exit()


if __name__ == "__main__":
    initialize()
    file = select_file()
    if file:
        file_name = Path(file).name
        make_index(file_name)
        ask(file_name)
    else:
        print("No files found")
        handle_exit()
