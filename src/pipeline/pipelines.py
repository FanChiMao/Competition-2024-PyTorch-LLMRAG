import os
import json

from tqdm import tqdm

from src.preprocess.loaders import KelvinPDFLoader, JonathanPDFLoader, TomPDFLoader, EdwardPDFLoader
from src.retrieve.retrievers import KelvinRetriever, JonathanRetriever, TomRetriever, EdwardRetriever
from src.preprocess.text_process import kelvin_preprocess, read_faq_data
from src.pipeline.base import BasePipeline

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains import ReduceDocumentsChain
from langchain_core.documents import Document


class KelvinPipeline(BasePipeline):
    def __init__(self, args, name='Kelvin'):
        super().__init__(args, name)

    def preprocess(self):
        # 產出insurance的corpus_dict
        source_path_insurance = os.path.join(self.args.source_path, 'insurance')  # 設定參考資料路徑
        PDFLoader_insurance = KelvinPDFLoader(source_path_insurance, self.insurance_pdf_pkl)
        self.corpus_dict_insurance = PDFLoader_insurance.load_data(use_pickle=self.use_pickle)

        # 產出finance的corpus_dict
        source_path_finance = os.path.join(self.args.source_path, 'finance')  # 設定參考資料路徑
        PDFLoader_finance = KelvinPDFLoader(source_path_finance, self.finance_pdf_pkl)
        self.corpus_dict_finance = PDFLoader_finance.load_data(use_pickle=self.use_pickle)

        # 產出faq的corpus_dict
        with open(os.path.join(self.args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
            key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
            key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

        self.corpus_dict_faq = {}
        for q_dict in self.qs_ref['questions']:
            if q_dict['category'] == 'faq':
                self.corpus_dict_faq.update({key: kelvin_preprocess(str(value)) for key, value in key_to_source_dict.items() if key in q_dict['source']})

    def retrieve(self):
        # 參考自得華分析結果，將不同類別的問題分別進行檢索
        Retriever_insurance = KelvinRetriever(self.corpus_dict_insurance, top_n=self.args.top_n)
        Retriever_finance = KelvinRetriever(self.corpus_dict_finance, top_n=self.args.top_n)
        Retriever_faq = KelvinRetriever(self.corpus_dict_faq, top_n=self.args.top_n)

        answer_dict = {"answers": []}  # 初始化字典

        for _, q_dict in enumerate(tqdm(self.qs_ref['questions'], desc=f"Answering questions by {self.name}")):
            if q_dict['category'] == 'finance':
                retrieved = Retriever_finance.retrieve(q_dict['query'], q_dict['source']) # 進行檢索
                answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved}) # 將結果加入字典
            elif q_dict['category'] == 'insurance':
                retrieved = Retriever_insurance.retrieve(q_dict['query'], q_dict['source'])
                answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
            elif q_dict['category'] == 'faq':
                retrieved = Retriever_faq.retrieve(q_dict['query'], q_dict['source'])
                answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
            else:
                raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

        return answer_dict


class JonathanPipeline(BasePipeline):
    def __init__(self, args, name='Jonathan'):
        super().__init__(args, name)
        self.reranker_args = self.config[self.name]['reranker']

    def preprocess(self):
        # 產出insurance的corpus_dict
        source_path_insurance = os.path.join(self.args.source_path, 'insurance')  # 設定參考資料路徑
        PDFLoader_insurance = KelvinPDFLoader(source_path_insurance, self.insurance_pdf_pkl)
        self.corpus_dict_insurance = PDFLoader_insurance.load_data(use_pickle=self.use_pickle)

        # 產出finance的corpus_dict
        source_path_finance = os.path.join(self.args.source_path, 'finance')  # 設定參考資料路徑
        PDFLoader_finance = KelvinPDFLoader(source_path_finance, self.finance_pdf_pkl)
        self.corpus_dict_finance = PDFLoader_finance.load_data(use_pickle=self.use_pickle)

        # 產出faq的corpus_dict
        with open(os.path.join(self.args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
            key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
            key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

        self.corpus_dict_faq = {}
        for q_dict in self.qs_ref['questions']:
            if q_dict['category'] == 'faq':
                self.corpus_dict_faq.update({key: str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']})

    def retrieve(self):
        # 參考自得華分析結果，將不同類別的問題分別進行檢索
        Retriever_insurance = JonathanRetriever(self.corpus_dict_insurance, top_n=self.args.top_n, reranker=self.reranker_args)
        Retriever_finance = JonathanRetriever(self.corpus_dict_finance, top_n=self.args.top_n, reranker=self.reranker_args)
        Retriever_faq = JonathanRetriever(self.corpus_dict_faq, top_n=self.args.top_n, reranker=self.reranker_args)

        answer_dict = {"answers": []}  # 初始化字典

        for _, q_dict in enumerate((tqdm(self.qs_ref['questions'], desc=f"Answering questions by {self.name}"))):

            if q_dict['category'] == 'finance':
                retrieved = Retriever_finance.retrieve(q_dict['query'], q_dict['source'])  # 進行檢索
                answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})  # 將結果加入字典
            elif q_dict['category'] == 'insurance':
                retrieved = Retriever_insurance.retrieve(q_dict['query'], q_dict['source'])
                answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
            elif q_dict['category'] == 'faq':
                retrieved = Retriever_faq.retrieve(q_dict['query'], q_dict['source'])
                answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
            else:
                raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

        return answer_dict


class TomPipeline(BasePipeline):
    def __init__(self, args, name='Tom'):
        super().__init__(args, name)

    def preprocess(self):
        # 產出insurance的corpus_dict
        source_path_insurance = os.path.join(self.args.source_path, 'insurance')  # 設定參考資料路徑
        PDFLoader_insurance = TomPDFLoader(source_path_insurance, self.insurance_pdf_pkl)
        self.corpus_dict_insurance = PDFLoader_insurance.load_data(use_pickle=self.use_pickle)

        # 產出finance的corpus_dict
        source_path_finance = os.path.join(self.args.source_path, 'finance')  # 設定參考資料路徑
        PDFLoader_finance = TomPDFLoader(source_path_finance, self.finance_pdf_pkl)
        self.corpus_dict_finance = PDFLoader_finance.load_data(use_pickle=self.use_pickle)

        # 產出faq的corpus_dict
        with open(os.path.join(self.args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
            key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
            key_to_source_dict = {int(key): read_faq_data(value) for key, value in key_to_source_dict.items()} # 整理讀取資料

        self.corpus_dict_faq = {}
        for q_dict in self.qs_ref['questions']:
            if q_dict['category'] == 'faq':
                self.corpus_dict_faq.update({key: str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']})

    def retrieve(self):
        # 參考自得華分析結果，將不同類別的問題分別進行檢索
        Retriever_insurance = TomRetriever(self.corpus_dict_insurance, top_n=self.args.top_n)
        Retriever_finance = TomRetriever(self.corpus_dict_finance, top_n=self.args.top_n)
        Retriever_faq = TomRetriever(self.corpus_dict_faq, top_n=self.args.top_n)

        answer_dict = {"answers": []}  # 初始化字典

        for _, q_dict in enumerate((tqdm(self.qs_ref['questions'], desc=f"Answering questions by {self.name}"))):
            if q_dict['category'] == 'finance':
                retrieved = Retriever_finance.retrieve(q_dict['query'], q_dict['source'])  # 進行檢索
                answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})  # 將結果加入字典
            elif q_dict['category'] == 'insurance':
                retrieved = Retriever_insurance.retrieve(q_dict['query'], q_dict['source'])
                answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
            elif q_dict['category'] == 'faq':
                retrieved = Retriever_faq.retrieve(q_dict['query'], q_dict['source'])
                answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
            else:
                raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

        return answer_dict


class EdwardPipeline(BasePipeline):
    def __init__(self, args, name='Edward'):
        super().__init__(args, name)


    def preprocess(self):
        # Generate vector datasets for each document
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
                    question = qa["question"] + "\n"
                    answer = ",".join(qa["answers"])
                    page_content = page_content + question + answer + "\n"

                document = Document(
                    page_content=page_content,
                    metadata={"source": fullpath, "pid": id},
                )
                docs = text_splitter.split_documents([document])
                all_docs.extend(docs)



        source_path_insurance = os.path.join(self.args.source_path, 'insurance')  # 設定參考資料路徑
        PDFLoader_insurance = EdwardPDFLoader(source_path_insurance, self.insurance_pdf_pkl)
        self.corpus_dict_insurance = PDFLoader_insurance.load_data(use_pickle=self.use_pickle)

        # 產出finance的corpus_dict
        source_path_finance = os.path.join(self.args.source_path, 'finance')  # 設定參考資料路徑
        PDFLoader_finance = EdwardPDFLoader(source_path_finance, self.finance_pdf_pkl)
        self.corpus_dict_finance = PDFLoader_finance.load_data(use_pickle=self.use_pickle)

        # 產出faq的corpus_dict
        with open(os.path.join(self.args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
            key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
            key_to_source_dict = {int(key): read_faq_data(value) for key, value in key_to_source_dict.items()} # 整理讀取資料

        self.corpus_dict_faq = {}
        for q_dict in self.qs_ref['questions']:
            if q_dict['category'] == 'faq':
                self.corpus_dict_faq.update({key: str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']})

    def retrieve(self):
        # 參考自得華分析結果，將不同類別的問題分別進行檢索
        Retriever_insurance = EdwardRetriever(self.corpus_dict_insurance, top_n=self.args.top_n)
        Retriever_finance = EdwardRetriever(self.corpus_dict_finance, top_n=self.args.top_n)
        Retriever_faq = EdwardRetriever(self.corpus_dict_faq, top_n=self.args.top_n)

        answer_dict = {"answers": []}  # 初始化字典

        for _, q_dict in enumerate((tqdm(self.qs_ref['questions'], desc=f"Answering questions by {self.name}"))):
            if q_dict['category'] == 'finance':
                retrieved = Retriever_finance.retrieve(q_dict['query'], q_dict['source'])  # 進行檢索
                answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})  # 將結果加入字典
            elif q_dict['category'] == 'insurance':
                retrieved = Retriever_insurance.retrieve(q_dict['query'], q_dict['source'])
                answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
            elif q_dict['category'] == 'faq':
                retrieved = Retriever_faq.retrieve(q_dict['query'], q_dict['source'])
                answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
            else:
                raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

        return answer_dict
