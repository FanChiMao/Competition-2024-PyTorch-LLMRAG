import os
import json
import yaml
from src.preprocess.loaders import BasePDFLoader
from src.retrieve.retrievers import BaseRetriever

class BasePipeline:
    def __init__(self, args, name="Base"):
        self.args = args
        self.name = name
        self.qs_ref = None
        self.corpus_dict_insurance = None
        self.corpus_dict_finance = None
        self.corpus_dict_faq = None
        self.load_yaml()
        
    def load_yaml(self):
        with open(self.args.yaml, 'r') as f:
            self.config = yaml.safe_load(f)
        self.use_pickle = self.config[self.name]['use_pickle']
        self.insurance_pdf_pkl = self.config[self.name]['insurance_pdf_pkl']
        self.finance_pdf_pkl = self.config[self.name]['finance_pdf_pkl']

    def preprocess(self):
        # 產出insurance的corpus_dict
        source_path_insurance = os.path.join(self.args.source_path, 'insurance')  # 設定參考資料路徑
        PDFLoader_insurance = BasePDFLoader(source_path_insurance, self.insurance_pdf_pkl)    
        self.corpus_dict_insurance = PDFLoader_insurance.load_data(use_pickle=self.use_pickle)

        # 產出finance的corpus_dict
        source_path_finance = os.path.join(self.args.source_path, 'finance')  # 設定參考資料路徑
        PDFLoader_finance = BasePDFLoader(source_path_finance, self.finance_pdf_pkl)
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
        Retriever_insurance = BaseRetriever(self.corpus_dict_insurance, top_n=self.args.top_n)
        Retriever_finance = BaseRetriever(self.corpus_dict_finance, top_n=self.args.top_n)
        Retriever_faq = BaseRetriever(self.corpus_dict_faq, top_n=self.args.top_n)

        answer_dict = {"answers": []}  # 初始化字典
        
        for q_dict in self.qs_ref['questions']:
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

    def run(self):
        with open(self.args.question_path, 'rb') as f:
            self.qs_ref = json.load(f)  # 讀取問題檔案

        # Preprocess
        self.preprocess()
        # Retrieve
        answer_dict = self.retrieve()        
        
        return answer_dict