import os
import faiss
import numpy as np
from typing import List
from src.embedding import Embedder
from src.data_loader import Document

class VectorIndex:
    """
    VectorIndex 利用 SentenceTransformer 计算文本嵌入，并使用 FAISS 构建向量索引。
    
    功能：
        - index_documents: 对输入的 Document 列表计算嵌入并建立索引。
        - search: 对查询文本计算嵌入，利用 FAISS 搜索最相似的向量，并返回对应的 Document 列表。
        - 支持索引持久化（写入和加载），满足本地部署需求。
    """
    
    def __init__(self, embedder: Embedder, vector_dim: int, index_path: str = None):
        self.embedder = embedder
        self.vector_dim = vector_dim
        self.index_path = index_path
        # 使用 FAISS 的 IndexFlatL2 作为基础索引
        self.index = faiss.IndexFlatL2(vector_dim)
        # 用于存储向量与文档的映射关系
        self.doc_mapping: List[Document] = []

        # 如果指定了索引持久化路径，尝试加载已有索引
        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            # 对于映射关系，这里需要额外加载，实际项目中可以考虑序列化存储
            mapping_path = index_path + ".mapping.npy"
            if os.path.exists(mapping_path):
                self.doc_mapping = list(np.load(mapping_path, allow_pickle=True))
    
    def index_documents(self, docs: List[Document]):
        """
        计算每个文档的嵌入向量，并将其添加到 FAISS 索引中，同时记录映射关系。
        """
        texts = [doc.text for doc in docs]
        embeddings = self.embedder.embed_text(texts)
        # 如果没有文本则 embeddings 可能为空，确保 shape 为 (0, vector_dim)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(0, self.vector_dim)
        # 添加向量到 FAISS 索引
        if embeddings.shape[0] > 0:
            self.index.add(embeddings)
        # 记录映射关系，保持顺序一致
        self.doc_mapping.extend(docs)
    
    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        对查询文本进行嵌入，利用 FAISS 搜索最相似的向量，并返回对应的 Document 列表。
        """
        # 如果 top_k 小于等于 0，则直接返回空列表
        if top_k <= 0:
            return []
        
        # 如果索引为空，直接返回空列表
        if self.index.ntotal == 0:
            return []
        
        query_embedding = self.embedder.embed_text([query])
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx in indices[0]:
            # FAISS 如果不足 top_k，会返回 -1 填充
            if idx == -1 or idx >= len(self.doc_mapping):
                continue
            results.append(self.doc_mapping[idx])
        return results


    def persist_index(self):
        """
        将 FAISS 索引和文档映射关系保存到磁盘。
        """
        if self.index_path:
            faiss.write_index(self.index, self.index_path)
            mapping_path = self.index_path + ".mapping.npy"
            np.save(mapping_path, np.array(self.doc_mapping, dtype=object))
        else:
            print("No index_path specified, skipping persist.")
