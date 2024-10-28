import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import warnings
warnings.filterwarnings("ignore")
import torch
import re
from collections import Counter, defaultdict

import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains import ReduceDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

import time

question_sheet_path = r'C:\EdwardLee\Project\AICUP2024\dataset\preliminary\questions_example.json'

ref_root_dir = r"C:\EdwardLee\Project\AICUP2024\reference"
ref_ids = [298,
            272,
            147,
            490,
            495]
text2vec_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 / sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

chunk_size = 50  # maximum ? characters in each chunk
chunk_overlap = 25
vector_db_type = "Chroma"  # FAISS / Chroma
answer_sheet = f"answer_{vector_db_type}_{chunk_size}_{chunk_overlap}.json"
answer_sheet_dir = os.path.join(r"C:\EdwardLee\Project\AICUP2024\dataset\preliminary", text2vec_model)
os.makedirs(answer_sheet_dir, exist_ok=True)
answer_sheet_path = os.path.join(answer_sheet_dir, answer_sheet)
vector_db_save_path = os.path.join("./vector_db", text2vec_model)

device = -1  # 0: GPU, -1: CPU  => huggyllama/llama-7b can only on CPU 😢 (at least 7*4 = 28 GB GPU RAM)
#-------------------------------------------------------------------------------------------------------------------

def check_filename_to_target_ref(filename, ref_ids):
    filename = os.path.splitext(filename)[0]
    for id in ref_ids:
        if filename == str(id):
            return True
    return False

def find_most_frequent_pid(topk_vecs):
    pids = [vec.metadata["pid"] for vec in topk_vecs]
    # 使用 Counter 計算每個 pid 出現的次數
    pid_counter = Counter(pids)

    # 找到出現次數最多的 pid
    most_common_pid, count = pid_counter.most_common(1)[0]
    if count == 1:
        return int(topk_vecs[0].metadata["pid"])
    return int(most_common_pid)

def find_weighted_frequent_pid(topk_vecs):
    pids = [vec.metadata["pid"] for vec in topk_vecs]
    
    # 使用 defaultdict 來存儲每個 pid 的權重分數
    pid_weights = defaultdict(float)

    # 計算權重：每次出現計算 1/排序分數
    pid_count = defaultdict(int)  # 用於計數每個 pid 的出現次數
    for index, pid in enumerate(pids, start=1):  # 對每個 pid 進行排序
        pid_count[pid] += 1
        pid_weights[pid] += 1 / index  # 將 1/n 加到該 pid 的權重上

    # 找出權重最高的 pid
    max_pid = max(pid_weights, key=pid_weights.get)
    max_weight = pid_weights[max_pid]

    return int(max_pid)

def find_rrf_pid(topk_vecs):
    k = len(topk_vecs)
    pids = [vec.metadata["pid"] for vec in topk_vecs]
    
    # 使用 defaultdict 來存儲每個 pid 的權重分數
    pid_weights = defaultdict(float)

    # 計算權重：每次出現計算 1/(排序+k) 分數
    for index, pid in enumerate(pids, start=1):  # 對每個 pid 進行排序
        pid_weights[pid] += 1 / (k + index)  # 將 1/(k+n) 加到該 pid 的權重上

    # 找出權重最高的 pid
    max_pid = max(pid_weights, key=pid_weights.get)
    max_weight = pid_weights[max_pid]

    return int(max_pid)

if __name__ == '__main__':

    print(">> Prepare text embedding model: %s\n"% text2vec_model)
    hf_embedder = HuggingFaceEmbeddings(model_name=text2vec_model)
    divider = "=" * 100
    print(divider + "\n")

    # 讀取 questions_example.json
    with open(question_sheet_path, 'r', encoding='utf-8') as file:
        questions_data = json.load(file)
        print(">> %s is loaded."% question_sheet_path) 
        print(divider + "\n")

    results = {'answers': []}
    for question in questions_data["questions"]:

        qid = question["qid"]
        ref_ids = question["source"]
        query = question["query"]
        category = question["category"]

        vector_db_save_name = f"qid_{qid}_chunk_size_{chunk_size}_overlap_{chunk_overlap}_{vector_db_type}_DB"
        db_path = os.path.join(vector_db_save_path, vector_db_save_name)
        os.makedirs(db_path, exist_ok=True)
        
        ref_dir = os.path.join(ref_root_dir, category)

        if not os.listdir(db_path):
            print(f">>> Split document into chunks\n")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            all_docs = []

            if category == 'faq':
                pid_map_content_file = os.listdir(ref_dir)[0]
                fullpath = os.path.join(ref_dir, pid_map_content_file)
                with open(fullpath, 'r', encoding='utf-8') as file:
                    pid_map_content_data = json.load(file)
                for id in ref_ids:
                    qas = pid_map_content_data[str(id)]
                    page_content = ""
                    for qa in qas:
                        question = qa["question"]+"\n"
                        answer = ",".join(qa["answers"])
                        page_content = page_content + question + answer + "\n"
                    
                    document = Document(
                        page_content=page_content,
                        metadata={"source": fullpath, "pid": id},
                    )
                    docs = text_splitter.split_documents([document])
                    all_docs.extend(docs)

            else:
                for pdf_file in os.listdir(ref_dir):
                    if check_filename_to_target_ref(pdf_file, ref_ids):
                        full_pdf_filepath = os.path.join(ref_dir, pdf_file)
                        pid, _ = os.path.splitext(pdf_file)
                        print(f">>> Load PDF from file: {full_pdf_filepath}\n")
                        loader = PyPDFLoader(full_pdf_filepath)
                        documents = loader.load()
                        for document in documents:
                            document.metadata['pid'] = pid
                        docs = text_splitter.split_documents(documents)
                        all_docs.extend(docs)
                
            print(f">>> Generate text embeddings for chunks\n")
            if vector_db_type == "FAISS":
                vectorstore = FAISS.from_documents(all_docs, hf_embedder)
                vectorstore.save_local(db_path)
            elif vector_db_type == "Chroma":
                vectorstore = Chroma.from_documents(all_docs, hf_embedder, persist_directory=db_path)
            else:
                raise ValueError("vector_db_type must be \"FAISS\" or \"Chroma\"\n")
            
            print(">>> Save vector db to: %s"% vector_db_save_name)
            print(divider + "\n")

        else:
            print(f">> Load previous vector database from \"{os.path.basename(db_path)}\"!\n")
            if vector_db_type == "FAISS":
                vectorstore = FAISS.load_local(db_path, hf_embedder, allow_dangerous_deserialization=True)
            elif vector_db_type == "Chroma":
                vectorstore = Chroma(persist_directory=db_path, embedding_function=hf_embedder)
            else:
                raise ValueError("vector_db_type must be \"FAISS\" or \"Chroma\"\n")
            print(divider + "\n")

        
        top_k = 5
        topk_vecs = vectorstore.similarity_search(query, top_k)
        if qid == 2:
            pass
        result = {
            'qid': qid,
            'retrieve': find_rrf_pid(topk_vecs)
        }
        results["answers"].append(result)
        
    
    # 將結果寫入 pred_retrieve.json
    with open(answer_sheet_path, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)

    print("計算完成，結果已保存到: %s"%answer_sheet_path)
    
