import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def chunk_document_str(text: str, chunk_size=500, overlap_size=100):
    """
    Split (chunk) the document text for avoiding the text over the max token length of the model.
    """
    # Only comparing the question text
    if "question" in text:  # FAQ
        data_list = eval(text)  # change to json format
        questions = [item['question'] for item in data_list if 'question' in item]
        return questions

    chunks = []
    stride = chunk_size - overlap_size
    if stride <= 0:
        raise ValueError(f"Overlap size must be smaller than chunk size, get {overlap_size}.")

    for i in range(0, len(text), stride):
        chunk_text = text[i:i + chunk_size]
        chunks.append(chunk_text)
    return chunks


class BaseReranker:
    def __init__(self, model_name, chunk_size=500, overlap_size=100):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.model = None
        self._use_reranker = False  # _use_reranker means use "reranker" or "embedder" to rerank

        if model_name in ["BAAI/bge-small-zh-v1.5", "BAAI/bge-base-zh-v1.5", "BAAI/bge-large-zh-v1.5", "BAAI/bge-m3"]:
            self.model = self._init_embedder_model()
            self._use_reranker = False
        elif model_name in ["BAAI/bge-reranker-base", "BAAI/bge-reranker-large"]:
            self.model = self._init_reranker_model()
            self._use_reranker = True
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    def _init_embedder_model(self):
        embedder = SentenceTransformer(self.model_name)
        return embedder

    def _init_reranker_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model.eval()
        return tokenizer, model


    def _rerank_with_reranker(self, query: str, top_docs: list[str]):
        tokenizer, model = self.model

        doc_to_max_similarity = []
        for doc in top_docs:
            chunk_texts = chunk_document_str(text=doc, chunk_size=self.chunk_size, overlap_size=self.overlap_size)

            query_chunk_pairs = [(query, chunk_text) for chunk_text in chunk_texts]
            inputs = tokenizer(query_chunk_pairs, padding=True, truncation=True, return_tensors="pt", max_length=self.chunk_size)
            chunk_similarities = model(**inputs, return_dict=True).logits.view(-1, )
            max_similarity = np.max(chunk_similarities.cpu().detach().numpy())
            doc_to_max_similarity.append(max_similarity)

        ranked_docs = sorted(zip(top_docs, doc_to_max_similarity), key=lambda x: x[1], reverse=True)

        return [doc[0] for doc in ranked_docs]


    def _rerank_with_embedder(self, query: str, top_docs: list[str]):
        query_embedding = self.model.encode(query, normalize_embeddings=True, convert_to_tensor=True)

        doc_to_max_similarity = []
        for doc in top_docs:
            chunk_texts = chunk_document_str(text=doc, chunk_size=self.chunk_size, overlap_size=self.overlap_size)
            chunk_embeddings = self.model.encode(chunk_texts, normalize_embeddings=True, convert_to_tensor=True)
            chunk_similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)

            max_similarity = np.max(chunk_similarities.cpu().numpy())
            doc_to_max_similarity.append(max_similarity)

        ranked_docs = sorted(zip(top_docs, doc_to_max_similarity), key=lambda x: x[1], reverse=True)

        return [doc[0] for doc in ranked_docs]

    def rerank_top_docs(self, query: str, top_docs: list[str]):
        if self._use_reranker:
            return self._rerank_with_reranker(query, top_docs)
        else:
            return self._rerank_with_embedder(query, top_docs)
