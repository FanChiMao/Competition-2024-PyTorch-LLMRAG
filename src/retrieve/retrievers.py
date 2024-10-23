import jieba
from rank_bm25 import BM25Plus
from src.retrieve.base import BaseRetriever
from src.preprocess.text_process import kelvin_preprocess

class KelvinRetriever(BaseRetriever):
    def __init__(self, corpus, top_n=1):
        super().__init__(corpus, top_n)

    def retrieve(self, query, source):
        # 文字前處理
        query = kelvin_preprocess(query)
        
        filtered_corpus = [self.corpus[int(file)] for file in source]

        # [Kelvin] 使用金管會 + 自定義字典
        jieba.load_userdict("./data/fsc_dict.txt")

        # [TODO] 可自行替換其他檢索方式，以提升效能
        # [Kelvin] 根據Tom建議使用BM25Plus演算法進行檢索
        # [Kelvin] 使用lcut_for_search進行分詞
        tokenized_corpus = [list(jieba.lcut_for_search(doc)) for doc in filtered_corpus]  # 將每篇文檔進行分詞        
        
        bm25 = BM25Plus(tokenized_corpus)  # 使用BM25演算法建立檢索模型
        tokenized_query = list(jieba.lcut_for_search(query))  # 將查詢語句進行分詞
        ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n= self.top_n)  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
        # 找回與最佳匹配文本相對應的檔案名
        res = [key for key, value in self.corpus.items() if value in ans]
        if self.top_n == 1:
            return res[0]
        return res[:self.top_n]  # 回傳檔案名