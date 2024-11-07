"""
Test the text splitter for langchain and custom implementation
"""
import os
import pdfplumber

from src.preprocess.text_process import edward_preprocess
from src.retrieve.reranker import chunk_document_str

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document


# # langchain implementation
# pdf_path = r"D:\Others\LLM_RAG\Competition-2024-PyTorch-LLMRAG\data/datasets/preliminary/finance/10.pdf"
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
# all_docs = []
# pid, _ = os.path.splitext(pdf_path)
# print(f">>> Load PDF from file: {pdf_path}\n")
# loader = PyPDFLoader(pdf_path)
# documents = loader.load()
# for document in documents:
#     document.metadata['pid'] = pid
#
#     # text preprocess
#     document.page_content = edward_preprocess(document.page_content)
#
# docs = text_splitter.split_documents(documents)
# all_docs.extend(docs)
# chunk_content = [doc.page_content for doc in all_docs]
# print(f">>> Split PDF into {len(all_docs)} chunks\n")
#
# # custom implementation
# from PyPDF2 import PdfReader
# reader = PdfReader(pdf_path)
# pdf_text = ""
# for page in reader.pages:
#     page_text = page.extract_text()  # Extract text from the page
#     if page_text:  # Ensure the page contains text
#         pdf_text += page_text + "\n"  # Add a newline for separation between pages
#
# chunked_texts = chunk_document_str(edward_preprocess(pdf_text), chunk_size=100, overlap_size=50, only_question=False)
#
# for str_langchain, str_custom in zip(chunk_content, chunked_texts):
#     if str_langchain != str_custom:
#         print(f"Langchain: {str_langchain}")
#         print(f"Custom: {str_custom}")

# ==================================================================================================================
# langchain implementation
import json
json_path = r"D:\Others\LLM_RAG\Competition-2024-PyTorch-LLMRAG\data\datasets\preliminary\faq\pid_map_content.json"
id = 283
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
all_docs = []
with open(json_path, 'r', encoding='utf-8') as file:
    pid_map_content_data = json.load(file)

qas = pid_map_content_data[str(id)]
page_content = ""
for qa in qas:
    question = qa["question"] + "\n"
    answer = ",".join(qa["answers"])
    page_content = page_content + question + answer + "\n"

document = Document(
    page_content=page_content,
    metadata={"source": json_path, "pid": id},
)
# text preprocess
texts = edward_preprocess(document.page_content)
document.page_content = texts
docs = text_splitter.split_documents([document])
all_docs.extend(docs)
langchain_chunk_content = [doc.page_content for doc in all_docs]

# custom implementation
custom_chunk_content = chunk_document_str(texts, chunk_size=100, overlap_size=50, only_question=False)
for str_langchain, str_custom in zip(langchain_chunk_content, custom_chunk_content):
    if str_langchain != str_custom:
        print(f"Langchain: {str_langchain}")
        print(f"Custom: {str_custom}")

for pid, qa_list in pid_map_content_data.items():
    str_ = ""
    for qa in qa_list:
        question = qa["question"] + "\n"
        answer = ",".join(qa["answers"])
        str_ = str_ + question + answer + "\n"

    test = edward_preprocess(str_)