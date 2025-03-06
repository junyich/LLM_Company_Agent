from typing import List, Tuple
from src.indexer import VectorIndex
from src.data_loader import Document

class VectorRetriever:
    """
    VectorRetriever 利用构建好的 VectorIndex 进行语义检索，
    返回 (Document, score) 对列表。
    
    假设 VectorIndex.search() 已返回 (Document, score) 对列表。
    """
    def __init__(self, vector_index: VectorIndex):
        self.vector_index = vector_index

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        if not query or top_k <= 0:
            return []
        results = self.vector_index.search(query, top_k=top_k)
        return results

# 测试代码（示例）
if __name__ == "__main__":
    from embedding.embedder import Embedder
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    vector_dim = 384
    from indexer.vector_index import VectorIndex, Document
    vector_index = VectorIndex(embedder=embedder, vector_dim=vector_dim, index_path=None)
    docs = [
        Document("机器学习是人工智能的重要分支。", metadata={"id": 1}),
        Document("深度学习在图像识别中有广泛应用。", metadata={"id": 2}),
        Document("自然语言处理可以理解和生成文本。", metadata={"id": 3})
    ]
    vector_index.index_documents(docs)
    retriever = VectorRetriever(vector_index=vector_index)
    query_text = "图像识别技术"
    results = retriever.retrieve(query_text, top_k=2)
    
    print("Vector检索结果：")
    for doc, score in results:
        print(f"Score: {score:.3f} - {doc}")
