import jieba_TW.jieba as jieba_tw
from rank_bm25 import BM25Plus
from src.retrieve.base import BaseRetriever
from src.retrieve.reranker import BaseReranker
from src.preprocess.text_process import kelvin_preprocess



class KelvinRetriever(BaseRetriever):
    def __init__(self, corpus, top_n=1):
        super().__init__(corpus, top_n)

    def retrieve(self, query, source):
        # 文字前處理
        query = kelvin_preprocess(query)
        
        filtered_corpus = [self.corpus[int(file)] for file in source]

        # [Kelvin] 使用金管會 + 自定義字典
        jieba_tw.dt.cache_file = 'jieba.cache.tw'
        jieba_tw.load_userdict("./data/custom_words/fsc_dict.txt")


        # [TODO] 可自行替換其他檢索方式，以提升效能
        # [Kelvin] 根據Tom建議使用BM25Plus演算法進行檢索
        # [Kelvin] 使用lcut_for_search進行分詞
        tokenized_corpus = [list(jieba_tw.lcut_for_search(doc)) for doc in filtered_corpus]  # 將每篇文檔進行分詞
        
        bm25 = BM25Plus(tokenized_corpus)  # 使用BM25演算法建立檢索模型
        tokenized_query = list(jieba_tw.lcut_for_search(query))  # 將查詢語句進行分詞
        ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n= self.top_n)  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
        # 找回與最佳匹配文本相對應的檔案名
        
        res = [key for value in ans for key, val in self.corpus.items() if val == value]

        return res[:self.top_n]  # 回傳檔案名


class JonathanRetriever(BaseRetriever):
    def __init__(self, corpus, top_n=1):
        super().__init__(corpus, top_n)
        self.use_reranker = True
        self.reranker_name = "BAAI/bge-small-zh-v1.5"
        self.reranker_model = None
        self.reranker_topn = max(3, top_n)
        self.chunk_size = 50
        self.overlap_size = 25

        if self.use_reranker:
            self.reranker_model = BaseReranker(self.reranker_name, self.chunk_size, self.overlap_size)

        # reranker (cross-encoder): BAAI/bge-reranker-base | BAAI/bge-reranker-large
        # embedder (bi-encoder): BAAI/bge-small-zh-v1.5 | BAAI/bge-base-zh-v1.5 | BAAI/bge-large-zh-v1.5 | BAAI/bge-m3

    def retrieve(self, query, source):
        # 文字前處理 (Query rewriter)
        # query = kelvin_preprocess(query)

        filtered_corpus = [self.corpus[int(file)] for file in source]

        # [Load corpus]
        jieba_tw.dt.cache_file = 'jieba.cache.tw'
        jieba_tw.load_userdict("data/custom_words/user_dict.txt")  # from Tom
        jieba_tw.load_userdict("data/custom_words/insurance_words.txt")  # from Jonathan

        # [BM25 first retrieval]
        tokenized_corpus = [list(jieba_tw.lcut_for_search(doc)) for doc in filtered_corpus]  # 將每篇文檔進行分詞
        bm25 = BM25Plus(tokenized_corpus)  # 使用BM25演算法建立檢索模型
        tokenized_query = list(jieba_tw.lcut_for_search(query))  # 將查詢語句進行分詞
        top_docs = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=self.reranker_topn)

        # [Reranker second sorting]
        if self.use_reranker and self.reranker_model is not None:
            reranked_top_docs = self.reranker_model.rerank_top_docs(query, top_docs)
            res = [key for value in reranked_top_docs for key, val in self.corpus.items() if val == value]
        else:
            res = [key for value in top_docs for key, val in self.corpus.items() if val == value]

        return res[:self.top_n]  # 回傳檔案名


class TomRetriever(BaseRetriever):
    def __init__(self, corpus, top_n=1):
        super().__init__(corpus, top_n)


    def retrieve(self, query, source):

        filtered_corpus = [self.corpus[int(file)] for file in source]

        # [Load corpus]
        jieba_tw.dt.cache_file = 'jieba.cache.tw'
        jieba_tw.load_userdict("data/custom_words/user_dict.txt")

        tokenized_corpus = [self._cut_words(doc) for doc in filtered_corpus]  # 將每篇文檔進行分詞

        bm25 = BM25Plus(tokenized_corpus)  # 使用BM25演算法建立檢索模型
        tokenized_query = self._cut_words(query)  # 將查詢語句進行分詞
        top_docs = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=self.top_n)  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項

        res = [key for value in top_docs for key, val in self.corpus.items() if val == value]

        return res[:self.top_n]  # 回傳檔案名


    @staticmethod
    def _cut_words(corpus: str) -> list[str]:
        result = []
        corpus = corpus.replace(" ", "")
        stop_word = {}.fromkeys([line.strip() for line in open(rb'./data/custom_words/stop_word.txt', encoding='utf-8')])

        for word in jieba_tw.lcut_for_search(corpus, HMM=False):
            if word != ' ' and word not in stop_word:
                result.append(word)
        return result
