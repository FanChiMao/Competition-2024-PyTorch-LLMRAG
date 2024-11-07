import os
import json

from tqdm import tqdm

from src.preprocess.loaders import KelvinPDFLoader, JonathanPDFLoader, TomPDFLoader, EdwardFileLoader
from src.retrieve.retrievers import KelvinRetriever, JonathanRetriever, TomRetriever, EdwardRetriever
from src.preprocess.text_process import kelvin_preprocess, edward_preprocess, read_faq_data
from src.pipeline.base import BasePipeline

from sentence_transformers import SentenceTransformer



class KelvinPipeline(BasePipeline):
    def __init__(self, args, name='Kelvin'):
        super().__init__(args, name)
        self.reranker_args = self.config['Reranker'] if self.config[name]['use_reranker'] else None

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
        Retriever_insurance = KelvinRetriever(self.corpus_dict_insurance, top_n=self.args.top_n, reranker=self.reranker_args)
        Retriever_finance = KelvinRetriever(self.corpus_dict_finance, top_n=self.args.top_n, reranker=self.reranker_args)
        Retriever_faq = KelvinRetriever(self.corpus_dict_faq, top_n=self.args.top_n, reranker=self.reranker_args)

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
        self.reranker_args = self.config['Reranker'] if self.config[name]['use_reranker'] else None

    def preprocess(self):
        # 產出insurance的corpus_dict
        source_path_insurance = os.path.join(self.args.source_path, 'insurance')  # 設定參考資料路徑
        PDFLoader_insurance = JonathanPDFLoader(source_path_insurance, self.insurance_pdf_pkl)
        self.corpus_dict_insurance = PDFLoader_insurance.load_data(use_pickle=self.use_pickle)

        # 產出finance的corpus_dict
        source_path_finance = os.path.join(self.args.source_path, 'finance')  # 設定參考資料路徑
        PDFLoader_finance = JonathanPDFLoader(source_path_finance, self.finance_pdf_pkl)
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
        self.reranker_args = self.config['Reranker'] if self.config[name]['use_reranker'] else None

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
        Retriever_insurance = TomRetriever(self.corpus_dict_insurance, top_n=self.args.top_n, reranker=self.reranker_args)
        Retriever_finance = TomRetriever(self.corpus_dict_finance, top_n=self.args.top_n, reranker=self.reranker_args)
        Retriever_faq = TomRetriever(self.corpus_dict_faq, top_n=self.args.top_n, reranker=self.reranker_args)

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
        # see ./data/pipeline.yml
        self.vector_db_args = self.config[self.name]['vector_db']
        self.embedding_args = self.config[self.name]['embedder']
        self.embedding_model = None
        self.insurance_db = None
        self.finance_db = None
        self.faq_db = None

        self._set_embedding_model()


    def _set_embedding_model(self):
        try:
            self.embedding_model = SentenceTransformer(self.embedding_args['embedder_name'])
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model \"{self.embedding_args['embedder_name']}\": {e}")


    def preprocess(self):  # Generate vector datasets (Chroma) for each document
        # 產出 insurance 的 vector database
        source_path_insurance = os.path.join(self.args.source_path, 'insurance')  # 設定參考資料路徑
        PDFLoader_insurance = EdwardFileLoader(source_dir=source_path_insurance, db_path=self.vector_db_args['insurance_db'], embedding_model=self.embedding_model)
        self.insurance_db = PDFLoader_insurance.load_vector_db_from_pdf(use_db=self.vector_db_args['use_db'], chunk_size=self.embedding_args['chunk_size'], overlap_size=self.embedding_args['overlap_size'])

        # 產出 finance 的 vector database
        source_path_finance = os.path.join(self.args.source_path, 'finance')  # 設定參考資料路徑
        PDFLoader_finance = EdwardFileLoader(source_dir=source_path_finance, db_path=self.vector_db_args['finance_db'], embedding_model=self.embedding_model)
        self.finance_db = PDFLoader_finance.load_vector_db_from_pdf(use_db=self.vector_db_args['use_db'], chunk_size=self.embedding_args['chunk_size'], overlap_size=self.embedding_args['overlap_size'])

        # 產出 faq 的 vector database
        source_path_faq = os.path.join(self.args.source_path, 'faq/pid_map_content.json')
        JSONLoader_faq = EdwardFileLoader(source_path_faq, db_path=self.vector_db_args['faq_db'], embedding_model=self.embedding_model)
        self.faq_db = JSONLoader_faq.load_vector_db_from_json(use_db=self.vector_db_args['use_db'], chunk_size=self.embedding_args['chunk_size'], overlap_size=self.embedding_args['overlap_size'])


    def retrieve(self):
        # 參考自得華分析結果，將不同類別的問題分別進行檢索
        insurance_db_name = os.path.basename(self.vector_db_args['insurance_db'])
        finance_db_name = os.path.basename(self.vector_db_args['finance_db'])
        faq_db_name = os.path.basename(self.vector_db_args['faq_db'])
        Retriever_insurance = EdwardRetriever(self.insurance_db, insurance_db_name, self.args.top_n, embedder=self.embedding_model)
        Retriever_finance = EdwardRetriever(self.finance_db, finance_db_name, top_n=self.args.top_n, embedder=self.embedding_model)
        Retriever_faq = EdwardRetriever(self.faq_db, faq_db_name, top_n=self.args.top_n, embedder=self.embedding_model)

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
