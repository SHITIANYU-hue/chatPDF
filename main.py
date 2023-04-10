import logging

logging.basicConfig(level=logging.CRITICAL)
import os
import sys
from pathlib import Path

import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from llama_index import GPTSimpleVectorIndex, LLMPredictor, ServiceContext, download_loader

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

FILES = "./files"
CACHE = f"{FILES}/.cache"


def init():
    if not os.path.exists(FILES):
        os.mkdir(FILES)
    if not os.path.exists(CACHE):
        os.mkdir(CACHE)


def handle_exit():
    print("\nGoodbye!\n")
    sys.exit(1)


def ask(file):
    print("👀 Loading...")

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.618, model_name="gpt-3.5-turbo", max_tokens=256))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=1024)
    # Check if file is in cache
    cache_file = os.path.join(CACHE, f"{Path(file).stem}.json")
    if os.path.exists(cache_file):
        index = GPTSimpleVectorIndex.load_from_disk(cache_file)
    else:
        PDFReader = download_loader("PDFReader")
        loader = PDFReader()
        documents = loader.load_data(file=Path(file))

        index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
        index.save_to_disk(cache_file)

    # clear the screen
    os.system("clear")

    print("✅ Ready! Let's start the conversation")
    print("ℹ️ Press Ctrl+C to exit")

    try:
        while True:
            prompt = input("\n😎 Prompt: ")
            if prompt == "exit":
                handle_exit()

            response = index.query(prompt)
            print()

            # transform response to string
            response = str(response)

            # if response starts with "\n", remove it
            if response.startswith("\n"):
                response = response[1:]

            print("👻 Response: " + response)
    except KeyboardInterrupt:
        handle_exit()


def select_file():
    os.system("clear")
    files = [file for file in os.listdir(FILES) if file.endswith(".pdf")]
    if len(files) == 0:
        return "file.pdf" if os.path.exists("file.pdf") else None
    print("📁 Select a file")
    for i, file in enumerate(files):
        print(f"{i+1}. {file}")
    print()

    try:
        possible_selections = [i for i in range(len(files) + 1)]
        selection = int(input("Enter a number, or 0 to exit: "))
        if selection == 0:
            handle_exit()
        elif selection not in possible_selections:
            select_file()
        else:
            file_path = os.path.abspath(os.path.join(FILES, files[selection - 1]))
    except ValueError:
        select_file()

    return file_path


if __name__ == "__main__":
    init()
    file = select_file()
    if file:
        ask(file)
    else:
        print("No files found")
        handle_exit()
