import os
import pickle
from tqdm import tqdm
import pdfplumber  # 用於從PDF文件中提取文字的工具
import numpy as np
from concurrent.futures import ProcessPoolExecutor

class BasePDFLoader:
    def __init__(self, source_dir, pickle_path=None, n_jobs=16):
        self.source_dir = source_dir
        self.pickle_path = pickle_path
        self.n_jobs = n_jobs
    
    def load_data(self, use_pickle=False):
        if use_pickle and self.pickle_path and os.path.exists(self.pickle_path):
            with open(self.pickle_path, 'rb') as f:
                corpus_dict = pickle.load(f)
            return corpus_dict
        else:
            corpus_dict = {}
            masked_file_ls = os.listdir(self.source_dir)  # 獲取資料夾中的檔案列表
            masked_file_ls = [os.path.join(self.source_dir, file) for file in masked_file_ls]
            pdf_files_split = np.array_split(masked_file_ls, self.n_jobs)
            
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                results = list(tqdm(executor.map(self.load_datas, pdf_files_split), total=self.n_jobs))
            
            for ele in results:
                corpus_dict.update(ele)
            
            if self.pickle_path:
                with open(self.pickle_path, 'wb') as f:
                    pickle.dump(corpus_dict, f)

            return corpus_dict
    
    def load_datas(self, pdf_files):
        corpus_dict = {int(os.path.basename(path).replace('.pdf', '')): self.read_pdf(path) for path in pdf_files}  # 讀取每個PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
        return corpus_dict

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
                pdf_text += text        

        pdf.close()  # 關閉PDF文件

        return pdf_text  # 返回萃取出的文本