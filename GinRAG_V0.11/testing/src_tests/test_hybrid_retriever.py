from typing import List

from src.data_loader import Document

# 导入相关检索模块
from src.embedding import Embedder
from src.indexer import VectorIndex
from src.retriever import VectorRetriever
from src.retriever import BM25Retriever
from src.retriever import HybridRetriever
from src.indexer.vector_index import VectorIndex

# 方便构造一个 VectorIndex 对象，若未持久化则 index_path 传 None
def build_vector_index(docs: List[Document], model_name="all-MiniLM-L6-v2", vector_dim=384):
    embedder = Embedder(model_name=model_name)
    vector_index = VectorIndex(embedder=embedder, vector_dim=vector_dim, index_path=None)
    vector_index.index_documents(docs)
    return vector_index

def build_retrievers(docs: List[Document]):
    # 构建向量检索器
    vector_index = build_vector_index(docs)
    vector_retriever = VectorRetriever(vector_index)
    # 构建 BM25 检索器
    bm25_retriever = BM25Retriever(docs)
    # 构建混合检索器
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
    return hybrid_retriever

# --------------------------
# Test Case 1: 空查询
# --------------------------
def test_empty_query():
    docs = [Document("machine learning is fun", metadata={"id": 1})]
    hr = build_retrievers(docs)
    results = hr.retrieve("", top_k=5)
    assert results == [], "Test Case 1 Failed: 空查询应返回空列表"
    print("Test Case 1 Passed: 空查询")

# --------------------------
# Test Case 2: top_k <= 0
# --------------------------
def test_top_k_le_zero():
    docs = [Document("machine learning is fun", metadata={"id": 1})]
    hr = build_retrievers(docs)
    results = hr.retrieve("machine learning", top_k=0)
    assert results == [], "Test Case 2 Failed: top_k<=0应返回空列表"
    print("Test Case 2 Passed: top_k<=0")

# --------------------------
# Test Case 3: 文档列表为空
# --------------------------
def test_no_documents():
    docs = []
    hr = build_retrievers(docs)
    results = hr.retrieve("machine learning", top_k=5)
    assert results == [], "Test Case 3 Failed: 空文档列表应返回空结果"
    print("Test Case 3 Passed: 空文档列表")

# --------------------------
# Test Case 4: 重复文档（相同 id），应去重
# --------------------------
def test_duplicate_removal_by_id():
    # 两个文档内容不同，但 metadata 中 id 相同，应视为重复
    docs = [
        Document("machine learning is fun", metadata={"id": 1}),
        Document("machine learning and deep learning", metadata={"id": 1})
    ]
    hr = build_retrievers(docs)
    results = hr.retrieve("machine learning", top_k=5)
    # 预期返回结果中只保留一份
    assert len(results) == 1, "Test Case 4 Failed: 重复文档应去重，只返回一条记录"
    print("Test Case 4 Passed: 重复文档去重")

# --------------------------
# Test Case 5: 两个不同文档，查询能返回两个结果
# --------------------------
def test_two_distinct_documents():
    docs = [
        Document("machine learning is fun", metadata={"id": 1}),
        Document("deep learning is powerful", metadata={"id": 2})
    ]
    hr = build_retrievers(docs)
    results = hr.retrieve("learning", top_k=5)
    # 期望返回至少两个结果
    assert len(results) == 2, "Test Case 5 Failed: 应返回两个不同的文档"
    print("Test Case 5 Passed: 返回两个不同文档")

# --------------------------
# Test Case 6: 模拟向量检索器返回空结果，BM25 返回有效结果
# --------------------------
def test_vector_empty_bm25_only():
    # 为模拟向量检索器返回空结果，构造一个空向量索引（不调用 index_documents）
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    vector_dim = 384
    # 构造一个空 VectorIndex
    empty_vector_index = VectorIndex(embedder=embedder, vector_dim=vector_dim, index_path=None)
    vector_retriever = VectorRetriever(empty_vector_index)
    # BM25 检索器依然使用有效文档
    docs = [Document("machine learning is fun", metadata={"id": 1})]
    bm25_retriever = BM25Retriever(docs)
    # 构造混合检索器
    hr = HybridRetriever(vector_retriever, bm25_retriever)
    results = hr.retrieve("machine learning", top_k=5)
    # 预期结果应包含 BM25 检索到的文档
    assert len(results) == 1, "Test Case 6 Failed: 当向量检索器为空时，应返回 BM25 的结果"
    print("Test Case 6 Passed: 向量检索为空时 BM25 返回结果")

# --------------------------
# Test Case 7: 模拟 BM25 返回空结果，向量检索返回有效结果
# --------------------------
def test_bm25_empty_vector_only():
    # 为模拟 BM25 返回空结果，构造一个文档，其文本不包含查询词
    docs = [Document("abcdefg", metadata={"id": 1})]
    # BM25 检索器会因查询词不匹配而返回空
    bm25_retriever = BM25Retriever(docs)
    # 向量检索器正常构造索引
    vector_index = build_vector_index(docs)
    vector_retriever = VectorRetriever(vector_index)
    hr = HybridRetriever(vector_retriever, bm25_retriever)
    results = hr.retrieve("machine learning", top_k=5)
    # 预期至少向量检索器返回结果（虽然语义可能较弱，但不为空）
    assert len(results) >= 1, "Test Case 7 Failed: 当 BM25 返回空时，应返回向量检索器的结果"
    print("Test Case 7 Passed: BM25返回空时 向量检索器返回结果")

# --------------------------
# Test Case 8: 两个检索器返回不完全重叠的结果
# --------------------------
def test_non_overlapping_results():
    # 构造文档，使得两个检索器可能返回不同结果（实际情况中可能有重叠，但我们构造模拟场景）
    docs = [
        Document("machine learning is fun", metadata={"id": 1}),
        Document("deep learning is powerful", metadata={"id": 2}),
        Document("natural language processing is interesting", metadata={"id": 3})
    ]
    hr = build_retrievers(docs)
    # 选择一个查询词，使得 BM25 和向量检索器各自更偏向于不同文档
    results = hr.retrieve("learning", top_k=2)
    # 合并后结果数应不超过文档总数，且至少包含一个文档
    assert len(results) > 0 and len(results) <= len(docs), "Test Case 8 Failed: 返回结果数异常"
    # 打印结果供观察
    print("Test Case 8 Passed: 非重叠结果测试")
    print("返回文档：", results)

# --------------------------
# Test Case 9: 重复文档（没有 id，但文本完全相同）应去重
# --------------------------
def test_duplicate_removal_by_text():
    docs = [
        Document("machine learning is fun", metadata={}),
        Document("machine learning is fun", metadata={})
    ]
    hr = build_retrievers(docs)
    results = hr.retrieve("machine learning", top_k=5)
    # 由于两个文档文本完全相同且没有 id 字段，采用文本作为 key，结果应只有一条
    assert len(results) == 1, "Test Case 9 Failed: 文本完全相同的文档应被去重"
    print("Test Case 9 Passed: 相同文本去重")


# --------------------------
# Test Case 10: 大量文档测试
# --------------------------
def test_large_document_set():
    # 构造 10 个不同的文档
    docs = [Document(f"document number {i} about machine learning", metadata={"id": i}) for i in range(1, 11)]
    hr = build_retrievers(docs)
    results = hr.retrieve("machine learning", top_k=10)
    # 期望返回结果数量等于 10 个（因为所有文档都相关且唯一）
    unique_ids = {doc.metadata["id"] for doc in results}
    assert len(unique_ids) == len(results) and len(results) == 10, "Test Case 10 Failed: 大量文档测试返回结果数不正确"
    print("Test Case 10 Passed: 大量文档测试")

# --------------------------
# 执行所有测试用例
# --------------------------


def testing_hybrid():
    test_empty_query()
    test_top_k_le_zero()
    test_no_documents()
    test_duplicate_removal_by_id()
    test_two_distinct_documents()
    test_vector_empty_bm25_only()
    test_bm25_empty_vector_only()
    test_non_overlapping_results()
    test_duplicate_removal_by_text()
    test_large_document_set()
    print("所有测试用例全部通过！")

if __name__ == "__main__":
    testing_hybrid()