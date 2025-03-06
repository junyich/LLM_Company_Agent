import unittest
from src.retriever.bm25_retriever import BM25Retriever
from src.data_loader import Document

class TestBM25Retriever(unittest.TestCase):
    def setUp(self):
        # 构造一组示例文档
        self.docs = [
            Document("Machine learning is an important branch of artificial intelligence.", metadata={"id": 1}),
            Document("Deep learning has widespread applications in image recognition.", metadata={"id": 2}),
            Document("Natural Language Processing can understand and generate text.", metadata={"id": 3}),
            Document("Artificial intelligence includes machine learning, deep learning, and natural language processing.", metadata={"id": 4}),
            Document("NLP, also known as natural language processing, is fascinating.", metadata={"id": 5}),
        ]
    
    def test_empty_documents(self):
        """测试空文档列表：无论查询如何，都应返回空列表。"""
        retriever = BM25Retriever(docs=[])
        results = retriever.retrieve("machine learning", top_k=3)
        self.assertEqual(len(results), 0, "Expected no results when document list is empty.")
    
    def test_empty_query(self):
        """测试空查询字符串：返回空列表。"""
        retriever = BM25Retriever(docs=self.docs)
        results = retriever.retrieve("", top_k=3)
        self.assertEqual(len(results), 0, "Expected no results for an empty query.")
    
    def test_no_matching_query(self):
        """测试查询中无匹配词：返回空列表。"""
        retriever = BM25Retriever(docs=self.docs)
        results = retriever.retrieve("biology", top_k=3)
        self.assertEqual(len(results), 0, "Expected no results for a query that matches nothing.")
    
    def test_multiple_matching_documents(self):
        """测试多个文档匹配：查询 'deep learning' 应返回包含该关键词的文档。"""
        retriever = BM25Retriever(docs=self.docs)
        results = retriever.retrieve("deep learning", top_k=3)
        self.assertGreater(len(results), 0, "Expected some results for query 'deep learning'.")
        # 检查返回结果中是否包含 metadata 中 id 为 2 和 4 的文档
        returned_ids = [doc.metadata.get("id") for doc, _ in results]
        self.assertIn(2, returned_ids, "Expected document with id 2 to be in the results.")
        self.assertIn(4, returned_ids, "Expected document with id 4 to be in the results.")
    
    def test_punctuation_and_case_sensitivity(self):
        """测试标点和大小写处理：查询 'nlp' 应能匹配包含 'NLP' 的文档。"""
        doc = Document("Natural Language Processing, also known as NLP! is fascinating.", metadata={"id": 6})
        retriever = BM25Retriever(docs=[doc])
        results = retriever.retrieve("nlp", top_k=1)
        self.assertEqual(len(results), 1, "Expected to find one matching document for query 'nlp'.")
        self.assertEqual(results[0][0].metadata.get("id"), 6, "Expected document with id 6 to match query 'nlp'.")
    
    def test_top_k_parameter(self):
        """测试 top_k 参数：查询 'machine' 时，即使匹配的文档超过 top_k，返回结果数量应不超过 top_k。"""
        retriever = BM25Retriever(docs=self.docs)
        results = retriever.retrieve("machine", top_k=2)
        self.assertLessEqual(len(results), 2, "Expected at most 2 results when top_k is set to 2.")
    
    def test_query_with_multiple_words(self):
        """测试多词查询：查询 'artificial intelligence machine learning' 应返回相关文档，且相关度较高的排在前面。"""
        retriever = BM25Retriever(docs=self.docs)
        results = retriever.retrieve("artificial intelligence machine learning", top_k=3)
        self.assertGreater(len(results), 0, "Expected some results for a multi-word query.")
        # 检查至少返回 id 为 1 和 4 的文档（二者均提及人工智能和机器学习）
        returned_ids = [doc.metadata.get("id") for doc, _ in results]
        self.assertIn(1, returned_ids, "Expected document with id 1 to be in the results.")
        self.assertIn(4, returned_ids, "Expected document with id 4 to be in the results.")
    
    def test_document_with_empty_text(self):
        """测试包含空文本的文档：空文本文档应不影响检索结果，且不会被返回。"""
        docs = self.docs + [Document("", metadata={"id": 7})]
        retriever = BM25Retriever(docs=docs)
        results = retriever.retrieve("machine learning", top_k=5)
        # 检查返回结果中不包含空文本文档（id 7）
        returned_ids = [doc.metadata.get("id") for doc, _ in results]
        self.assertNotIn(7, returned_ids, "Expected document with id 7 (empty text) not to be in the results.")
    
    def test_special_characters_in_query(self):
        """测试查询中包含特殊字符：查询 'NLP!!!' 应正确匹配包含 'NLP' 的文档。"""
        doc = Document("NLP is a key part of modern AI.", metadata={"id": 8})
        retriever = BM25Retriever(docs=[doc])
        results = retriever.retrieve("NLP!!!", top_k=1)
        self.assertEqual(len(results), 1, "Expected to find one matching document for query 'NLP!!!'.")
        self.assertEqual(results[0][0].metadata.get("id"), 8, "Expected document with id 8 to match query 'NLP!!!'.")

def bm25_test():
    unittest.main()

if __name__ == "__main__":
    bm25_test()
