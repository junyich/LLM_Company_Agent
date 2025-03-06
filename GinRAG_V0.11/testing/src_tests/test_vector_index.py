import os
import tempfile
import unittest
import numpy as np

# python -m indexer.test_vector_index

# 导入待测试模块中的 VectorIndex 和 Document
from src.indexer import VectorIndex
from src.data_loader import Document

# 定义一个 DummyEmbedder 用于测试，保证嵌入结果可控
class DummyEmbedder:
    def __init__(self, vector_dim: int):
        self.vector_dim = vector_dim

    def embed_text(self, texts, batch_size: int = 32) -> np.ndarray:
        # 如果 texts 为空，返回形状 (0, vector_dim) 的数组
        if not texts:
            return np.empty((0, self.vector_dim), dtype=np.float32)
        # 返回的向量由文本长度构成，每个文本返回一个向量：[len(text)] * vector_dim
        return np.array([[len(text)] * self.vector_dim for text in texts], dtype=np.float32)

class TestVectorIndex(unittest.TestCase):

    def setUp(self):
        # 创建一个临时目录用于持久化测试
        self.temp_dir = tempfile.TemporaryDirectory()
        self.index_path = os.path.join(self.temp_dir.name, "faiss.index")
        self.vector_dim = 10  # 设定一个较小的维度用于测试
        self.embedder = DummyEmbedder(self.vector_dim)
        self.vector_index = VectorIndex(embedder=self.embedder, vector_dim=self.vector_dim, index_path=self.index_path)

    def tearDown(self):
        # 清理临时目录
        self.temp_dir.cleanup()

    def test_index_documents_empty(self):
        # 测试传入空文档列表
        self.vector_index.index_documents([])
        # FAISS 索引中应该没有数据
        self.assertEqual(self.vector_index.index.ntotal, 0)
        self.assertEqual(len(self.vector_index.doc_mapping), 0)

    def test_search_no_docs(self):
        # 当索引为空时，search 应该返回空列表
        results = self.vector_index.search("any query", top_k=5)
        self.assertEqual(len(results), 0)

    def test_index_and_search(self):
        # 构造三个文档：文本长度决定了向量的值
        doc1 = Document("a", metadata={"id": 1})    # 长度 1
        doc2 = Document("aa", metadata={"id": 2})   # 长度 2
        doc3 = Document("aaa", metadata={"id": 3})  # 长度 3
        docs = [doc1, doc2, doc3]
        self.vector_index.index_documents(docs)
        
        # 检查 FAISS 中向量数是否正确
        self.assertEqual(self.vector_index.index.ntotal, 3)
        self.assertEqual(len(self.vector_index.doc_mapping), 3)

        # 查询与 "a" 的长度相同，期望返回 doc1
        results = self.vector_index.search("a", top_k=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].metadata["id"], 1)

        # 查询 top_k 大于文档数，应只返回3个有效文档
        results = self.vector_index.search("aaa", top_k=5)
        self.assertEqual(len(results), 3)

    def test_persist_and_reload_index(self):
        # 构造示例文档并索引
        docs = [
            Document("test1", metadata={"id": 101}),
            Document("longer text", metadata={"id": 102})
        ]
        self.vector_index.index_documents(docs)
        # 持久化索引
        self.vector_index.persist_index()

        # 新建一个 VectorIndex 实例，并加载之前保存的索引和映射关系
        new_vector_index = VectorIndex(embedder=self.embedder, vector_dim=self.vector_dim, index_path=self.index_path)
        # 确保映射关系正确加载
        self.assertEqual(len(new_vector_index.doc_mapping), len(docs))

        # 对同一个查询，两个索引返回的结果应该一致
        query = "test"
        results_old = self.vector_index.search(query, top_k=2)
        results_new = new_vector_index.search(query, top_k=2)
        # 比较文档 metadata id
        ids_old = [doc.metadata["id"] for doc in results_old]
        ids_new = [doc.metadata["id"] for doc in results_new]
        self.assertEqual(ids_old, ids_new)

    # 新增测试用例1：测试多次调用 index_documents 累加效果
    def test_multiple_indexing(self):
        doc1 = Document("hello", metadata={"id": 1})
        doc2 = Document("world", metadata={"id": 2})
        self.vector_index.index_documents([doc1])
        self.assertEqual(self.vector_index.index.ntotal, 1)
        self.assertEqual(len(self.vector_index.doc_mapping), 1)
        self.vector_index.index_documents([doc2])
        self.assertEqual(self.vector_index.index.ntotal, 2)
        self.assertEqual(len(self.vector_index.doc_mapping), 2)

    # 新增测试用例2：搜索空查询字符串
    def test_search_empty_query(self):
        # 添加一个空文本的文档
        doc_empty = Document("", metadata={"id": 100})
        doc_nonempty = Document("nonempty", metadata={"id": 101})
        self.vector_index.index_documents([doc_empty, doc_nonempty])
        # 搜索空字符串，空文档的向量应该全部为0
        results = self.vector_index.search("", top_k=2)
        # 预期第一个结果应为 doc_empty（id 100）
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].metadata["id"], 100)

    # 新增测试用例3：相同文档重复索引
    def test_index_same_document_multiple_times(self):
        doc = Document("repeat", metadata={"id": 200})
        self.vector_index.index_documents([doc, doc])
        self.assertEqual(self.vector_index.index.ntotal, 2)
        self.assertEqual(len(self.vector_index.doc_mapping), 2)
        results = self.vector_index.search("repeat", top_k=5)
        # 两个重复的文档都应该出现在结果中
        self.assertEqual(len(results), 2)

    # 新增测试用例4：查询时 top_k 设置为 0，应返回空列表
    def test_search_top_k_zero(self):
        doc = Document("sample", metadata={"id": 300})
        self.vector_index.index_documents([doc])
        results = self.vector_index.search("sample", top_k=0)
        self.assertEqual(len(results), 0)

    # 新增测试用例5：查询返回的文档数不超过实际索引的数量
    def test_search_top_k_exceeds_total(self):
        docs = [
            Document("one", metadata={"id": 1}),
            Document("two", metadata={"id": 2})
        ]
        self.vector_index.index_documents(docs)
        # 设置 top_k 大于文档数，返回的结果数量应等于文档数
        results = self.vector_index.search("o", top_k=10)
        self.assertEqual(len(results), 2)

    # 新增测试用例6：检查文档映射关系的顺序完整性
    def test_document_mapping_integrity(self):
        docs = [
            Document("first", metadata={"id": 10}),
            Document("second", metadata={"id": 20}),
            Document("third", metadata={"id": 30})
        ]
        self.vector_index.index_documents(docs)
        # 检查映射顺序与输入顺序一致
        for i, doc in enumerate(docs):
            self.assertEqual(self.vector_index.doc_mapping[i].metadata["id"], doc.metadata["id"])

def test_vector():
    unittest.main()

if __name__ == '__main__':
    unittest.main()
