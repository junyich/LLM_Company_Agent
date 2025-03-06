from typing import List, Tuple
from src.data_loader import Document
from src.retriever.vector_retriever import VectorRetriever
from src.retriever.bm25_retriever import BM25Retriever
import hashlib

class HybridRetriever:
    """
    HybridRetriever 结合向量检索和 BM25 检索，并采用动态加权融合策略，
    返回最终排序后的 Document 列表。
    """
    def __init__(self, vector_retriever: VectorRetriever, bm25_retriever: BM25Retriever,
                 weight_vector: float = 0.7, weight_bm25: float = 0.3):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.weight_vector = weight_vector
        self.weight_bm25 = weight_bm25

    def get_doc_key(self, doc: Document) -> str:
        """
        返回文档的唯一标识：
          - 优先尝试从 metadata 中获取 'id'
          - 如果不存在，则计算 doc.text 的 MD5 哈希
        """
        if "id" in doc.metadata:
            return str(doc.metadata["id"])
        else:
            return hashlib.md5(doc.text.encode("utf-8")).hexdigest()

    def expand_query(self, query: str) -> str:
        """
        简单的查询扩展示例，可以根据领域词典进行同义词扩展或上下文补充。
        此处仅作示例，如果查询中包含“方案”，则扩展查询词。
        """
        if "方案" in query:
            return query + " 实施计划 计划书"
        return query

    def rerank_results(self, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """
        占位函数：可利用更复杂的重排序模型（例如跨编码器）对候选结果进行二次排序。
        目前直接返回原结果。
        """
        return results

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        对查询文本分别调用向量检索和 BM25 检索，经过动态加权融合后返回最终的 Document 列表。
        """
        if not query or top_k <= 0:
            return []

        # 查询扩展（Prompt优化）
        expanded_query = self.expand_query(query)

        # 分别调用两个检索器，均要求返回 (doc, score) 对
        vector_results = self.vector_retriever.retrieve(expanded_query, top_k=top_k)
        bm25_results = self.bm25_retriever.retrieve(expanded_query, top_k=top_k)

        # 融合结果：采用字典以文档唯一标识为键
        combined_scores = {}

        # 融合向量检索结果：处理可能非 tuple 的情况
        for item in vector_results:
            if isinstance(item, tuple):
                doc, score = item
            else:
                doc, score = item, 0.0
            key = self.get_doc_key(doc)
            if key not in combined_scores:
                combined_scores[key] = {"doc": doc, "vector_score": score, "bm25_score": 0.0}
            else:
                combined_scores[key]["vector_score"] = max(combined_scores[key]["vector_score"], score)

        # 融合 BM25 检索结果（这里假设 BM25 检索器始终返回 tuple）
        for doc, score in bm25_results:
            key = self.get_doc_key(doc)
            if key not in combined_scores:
                combined_scores[key] = {"doc": doc, "vector_score": 0.0, "bm25_score": score}
            else:
                combined_scores[key]["bm25_score"] = max(combined_scores[key]["bm25_score"], score)

        # 动态加权融合：综合得分 = weight_vector * vector_score + weight_bm25 * bm25_score
        fused_results = []
        for entry in combined_scores.values():
            fused_score = self.weight_vector * entry["vector_score"] + self.weight_bm25 * entry["bm25_score"]
            fused_results.append((entry["doc"], fused_score))

        # 可选：进行二次重排序
        fused_results = self.rerank_results(fused_results)

        # 根据综合得分降序排序
        fused_results.sort(key=lambda x: x[1], reverse=True)

        # 返回排序后的前 top_k 个文档（只返回 Document 对象）
        final_docs = [doc for doc, score in fused_results[:top_k]]
        return final_docs
