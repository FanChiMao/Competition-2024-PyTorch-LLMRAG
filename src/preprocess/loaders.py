import os
import re
import json
import pdfplumber
from PyPDF2 import PdfReader
import pytesseract
import chromadb

from tqdm import tqdm

from src.preprocess.base import BasePDFLoader
from src.preprocess.text_process import kelvin_preprocess, edward_preprocess
from src.retrieve.reranker import chunk_document_str


class KelvinPDFLoader(BasePDFLoader):
    def __init__(self, source_dir, pickle_path=None, n_jobs=16):
        super().__init__(source_dir, pickle_path, n_jobs)
        if pickle_path is not None and not os.path.exists(os.path.dirname(pickle_path)):
            os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

    @staticmethod
    def read_pdf(pdf_loc, page_infos: list = None):
        pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

        # [TODO]: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理
        # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
        pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
        pdf_text = ''
        for _, page in enumerate(pages):  # 迴圈遍歷每一頁
            text = page.extract_text()  # 提取頁面的文本內容

            if text:
                text = kelvin_preprocess(text)
                pdf_text += text

        pdf.close()  # 關閉PDF文件

        return pdf_text  # 返回萃取出的文本


class JonathanPDFLoader(BasePDFLoader):
    def __init__(self, source_dir, pickle_path=None, n_jobs=16):
        super().__init__(source_dir, pickle_path, n_jobs)
        if pickle_path is not None and not os.path.exists(os.path.dirname(pickle_path)):
            os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

    @staticmethod
    def read_pdf(pdf_loc, page_infos: list = None):
        pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

        pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
        pdf_text = ''
        for _, page in enumerate(pages):  # 迴圈遍歷每一頁
            text = page.extract_text()  # 提取頁面的文本內容

            if text:
                pdf_text += text

        pdf.close()  # 關閉PDF文件

        pdf_text = pdf_text.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")

        return pdf_text  # 返回萃取出的文本


class TomPDFLoader(BasePDFLoader):
    def __init__(self, source_dir, pickle_path=None, n_jobs=16):
        super().__init__(source_dir, pickle_path, n_jobs)
        if pickle_path is not None and not os.path.exists(os.path.dirname(pickle_path)):
            os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

    @staticmethod
    def read_pdf(pdf_loc, page_infos: list = None):
        pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

        # TODO: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理

        # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
        pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
        pdf_text = ''
        for _, page in enumerate(pages):  # 迴圈遍歷每一頁
            text = page.extract_text()  # 提取頁面的文本內容
            if text:
                text = ''.join(text.splitlines())
                pdf_text += text
            # 20241025 Jonathan OCR 失敗 先省略 😢
            # else:
            #     # 使用 OCR 方式擷取頁面文字
            #     image = page.to_image(resolution=500)  # 提取頁面的文本畫面
            #     ocr_text = pytesseract.image_to_string(image.original, lang='chi_tra', config="--psm 12")
            #     if ocr_text:
            #         ocr_text = ''.join(ocr_text.replace(" ", "").splitlines())
            #         pdf_text += ocr_text

            # only use OCR
            # image = page.to_image(resolution=500)  # 提取頁面的文本畫面
            # ocr_text = pytesseract.image_to_string(image.original, lang='chi_tra', config="--psm 12")
            # if ocr_text:
            #     ocr_text = ''.join(ocr_text.replace(" ", "").splitlines())
            #     pdf_text += ocr_text

        pdf.close()  # 關閉PDF文件

        return pdf_text  # 返回萃取出的文本


class EdwardFileLoader(BasePDFLoader):
    def __init__(self, source_dir, pickle_path=None, n_jobs=16, db_path=None, embedding_model=None):
        super().__init__(source_dir, pickle_path, n_jobs)
        self.db_path = db_path
        self.embedding_model = embedding_model

        if self.embedding_model is None:
            raise ValueError("Embedding model is required")

        if pickle_path is not None and not os.path.exists(os.path.dirname(pickle_path)):
            os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

        if db_path is not None and not os.path.exists(os.path.dirname(db_path)):
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

        os.makedirs(self.db_path, exist_ok=True)


    @staticmethod
    def read_pdf(pdf_loc, page_infos: list = None):
        reader = PdfReader(pdf_loc)
        pdf_text = ""
        for page in reader.pages:
            page_text = page.extract_text()  # Extract text from the page
            if page_text:  # Ensure the page contains text
                pdf_text += page_text + "\n"  # Add a newline for separation between pages

        return pdf_text  # 返回萃取出的文本


    def load_vector_db_from_pdf(self, use_db=False, chunk_size=500, overlap_size=250):
        db_name = os.path.basename(self.db_path)
        if use_db and self.db_path and os.listdir(self.db_path):
            chroma_client = chromadb.PersistentClient(path=self.db_path)
            vector_db = chroma_client.get_collection(name=db_name)
            return vector_db

        else:
            masked_file_ls = os.listdir(self.source_dir)
            masked_file_ls = [os.path.join(self.source_dir, file) for file in masked_file_ls]

            # create vector database
            chroma_client = chromadb.PersistentClient(path=self.db_path)
            vector_db = chroma_client.get_or_create_collection(name=db_name)

            # load all pdf documents
            for i, pdf_path in enumerate(tqdm(masked_file_ls)):
                ids = []
                texts = []
                metadatas = []
                embeddings = []
                full_text = self.read_pdf(pdf_path)
                full_text = edward_preprocess(full_text)
                chunked_texts = chunk_document_str(full_text, chunk_size=chunk_size, overlap_size=overlap_size, only_question=False)

                file_id = os.path.splitext(os.path.basename(pdf_path))[0]
                for chunk_idx, chunk_text in enumerate(chunked_texts):
                    chunk_id = f"file_{file_id}_chunk_{chunk_idx}"
                    embedding = self.embedding_model.encode(chunk_text).tolist()

                    ids.append(chunk_id)  # Ensure a unique ID for each chunk
                    texts.append(chunk_text)
                    embeddings.append(embedding)
                    metadatas.append({"category": str(db_name), "source": pdf_path, "pid": int(file_id)})

                if len(texts) == 0:  # Skip empty documents
                    continue

                vector_db.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)

            return vector_db


    def load_vector_db_from_json(self, use_db=False, chunk_size=500, overlap_size=250):
        db_name = os.path.basename(self.db_path)  # faq
        if use_db and self.db_path and os.listdir(self.db_path):
            chroma_client = chromadb.PersistentClient(path=self.db_path)
            vector_db = chroma_client.get_collection(name=db_name)
            return vector_db

        else:
            # create vector database
            chroma_client = chromadb.PersistentClient(path=self.db_path)
            vector_db = chroma_client.get_or_create_collection(name=db_name)

            with open(self.source_dir, 'rb') as f_s:
                pid_map_content_data = json.load(f_s)  # 讀取參考資料文件

            for pid, qa_list in tqdm(pid_map_content_data.items()):
                page_content = ""
                for qa in qa_list:
                    question = qa["question"] + "\n"
                    answer = ",".join(qa["answers"])
                    page_content = page_content + question  + answer + "\n"

                page_content = edward_preprocess(page_content)

                ids = []
                texts = []
                metadatas = []
                embeddings = []
                chunked_texts = chunk_document_str(page_content, chunk_size=chunk_size, overlap_size=overlap_size, only_question=False)

                for chunk_idx, chunk_text in enumerate(chunked_texts):
                    chunk_id = f"qa_{pid}_chunk_{chunk_idx}"
                    embedding = self.embedding_model.encode(chunk_text).tolist()
                    embeddings.append(embedding)
                    ids.append(chunk_id)  # Ensure a unique ID for each chunk
                    texts.append(chunk_text)
                    metadatas.append({"category": str(db_name), "source": self.source_dir, "pid": int(pid)})

                if len(texts) == 0:  # Skip empty documents
                    continue

                vector_db.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)

            return vector_db
