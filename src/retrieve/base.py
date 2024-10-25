import jieba
from rank_bm25 import BM25Okapi

class BaseRetriever:
    def __init__(self, corpus, top_n=1):
        self.corpus = corpus
        self.top_n = top_n
    
    def retrieve(self, query, source):
        filtered_corpus = [self.corpus[int(file)] for file in source]

        # [TODO] 可自行替換其他檢索方式，以提升效能
        tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]  # 將每篇文檔進行分詞
        bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
        tokenized_query = list(jieba.cut_for_search(query))  # 將查詢語句進行分詞
        ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n = self.top_n)  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
        
        # 找回與最佳匹配文本相對應的檔案名
        res = [key for value in ans for key, val in self.corpus.items() if val == value]
        if self.top_n == 1:
            return res[0]
        return res[:self.top_n]  # 回傳檔案名
    