import re
import jieba
from typing import List, Tuple
from src.data_loader import Document
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> List[str]:
    """
    自适应分词函数：
    - 如果文本为纯 ASCII（英文），使用正则表达式分词，且转为小写；
    - 否则使用 jieba 的搜索模式分词，过滤空白。
    """
    if text.isascii():
        return re.findall(r'\w+', text.lower())
    else:
        return [token.strip() for token in jieba.lcut_for_search(text) if token.strip()]

class BM25Retriever:
    """
    BM25Retriever 利用 BM25 算法对文档进行关键词匹配检索，
    返回 (Document, score) 对的列表。
    """
    def __init__(self, docs: List[Document]):
        self.docs = docs
        if not docs:
            self.tokenized_corpus = []
            self.bm25 = None
        else:
            self.tokenized_corpus = [tokenize(doc.text) for doc in docs]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        对查询文本进行 BM25 检索，返回 (Document, score) 对列表。
        """
        if not query or top_k <= 0 or self.bm25 is None:
            return []
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        if all(score == 0 for score in scores):
            return []
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = [(self.docs[i], scores[i]) for i in top_indices]
        return results

# 可选：直接运行该模块进行简单测试
if __name__ == "__main__":
    docs = [
        Document("Machine learning is an important branch of artificial intelligence.", metadata={"id": 1}),
        Document("Deep learning has widespread applications in image recognition.", metadata={"id": 2}),
        Document("Natural Language Processing can understand and generate text.", metadata={"id": 3}),
        Document("Artificial intelligence includes machine learning, deep learning and natural language processing.", metadata={"id": 4}),
        Document("NLP, also known as natural language processing, is fascinating.", metadata={"id": 5}),
    ]
    retriever = BM25Retriever(docs=docs)
    results = retriever.retrieve("deep learning", top_k=3)
    for doc, score in results:
        print(f"Score: {score:.3f} - Document ID: {doc.metadata.get('id')}")
