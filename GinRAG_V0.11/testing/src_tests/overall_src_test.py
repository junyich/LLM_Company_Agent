import os
import tempfile


def test_normal_flow():
    # 创建临时目录模拟 markdown 文件夹
    with tempfile.TemporaryDirectory() as temp_dir:
        md_folder = os.path.join(temp_dir, "markdown_docs")
        os.mkdir(md_folder)
        
        # 写入 3 个简单的 Markdown 文件
        file_contents = [
            "### Doc1\n机器学习是人工智能的重要分支。",
            "---\ntitle: Doc2\nauthor: Test\n---\n深度学习在图像识别中有广泛应用。",
            "Doc3 内容：自然语言处理可以理解和生成文本。"
        ]
        filenames = ["Doc1.md", "Doc2.md", "Doc3.md"]
        for fname, content in zip(filenames, file_contents):
            with open(os.path.join(md_folder, fname), "w", encoding="utf-8") as f:
                f.write(content)
        
        # 1. 加载文档
        from src.data_loader.markdown_loader import MarkdownLoader
        loader = MarkdownLoader(md_folder)
        documents = loader.load_data()
        assert len(documents) == 3, "应加载 3 个文档"
        
        # 2. 分段处理
        from src.preprocessor.text_splitter import HierarchicalTextSplitter
        splitter = HierarchicalTextSplitter(chunk_sizes=[2048, 512, 256], chunk_overlap=50)
        processed_docs = []
        for doc in documents:
            chunks = splitter.split_text(doc.text)
            for chunk in chunks:
                processed_docs.append(doc.__class__(text=chunk, metadata=doc.metadata))
        assert len(processed_docs) > 0, "分段后应获得至少 1 个文本块"
        
        # 3. 构建向量索引
        from src.embedding.embedder import Embedder
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        vector_dim = 384
        from src.indexer.vector_index import VectorIndex
        vector_index = VectorIndex(embedder=embedder, vector_dim=vector_dim, index_path=None)
        vector_index.index_documents(processed_docs)
        assert vector_index.index.ntotal > 0, "FAISS 索引中应有数据"
        
        # 4. 构建检索器
        from src.retriever.vector_retriever import VectorRetriever
        from src.retriever.bm25_retriever import BM25Retriever
        from src.retriever.hybrid_retriever import HybridRetriever
        vector_retriever = VectorRetriever(vector_index)
        bm25_retriever = BM25Retriever(processed_docs)
        hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
        
        # 5. 模拟查询
        query = "图像识别"
        results = hybrid_retriever.retrieve(query, top_k=3)
        # 根据测试文件内容，至少应返回一条结果
        assert len(results) > 0, "检索结果不应为空"
        
        # 6. 构造 prompt 并调用 LLM（此处可选择模拟 LLM 返回）
        combined_context = "\n".join([doc.text for doc in results])
        prompt = f"基于以下内容回答问题：\n{combined_context}\n问题：{query}\n回答："
        
        from src.llm_interface.local_llm import LocalLLM
        # 为测试可以用一个模拟 LLM 或真实调用
        llm = LocalLLM(model_name="deepseek-r1")
        response = llm.complete(prompt)
        print("测试生成回答：", response.text)

def test_edge_cases():
    # Edge Case 1：空文件夹
    empty_dir = tempfile.TemporaryDirectory().name
    from src.data_loader.markdown_loader import MarkdownLoader
    loader = MarkdownLoader(empty_dir)
    documents = loader.load_data()
    assert len(documents) == 0, "空文件夹应加载 0 个文档"
    
    # Edge Case 2：文档内容为空
    with tempfile.TemporaryDirectory() as temp_dir:
        md_folder = os.path.join(temp_dir, "markdown_docs")
        os.mkdir(md_folder)
        with open(os.path.join(md_folder, "empty.md"), "w", encoding="utf-8") as f:
            f.write("")
        loader = MarkdownLoader(md_folder)
        documents = loader.load_data()
        # 根据实现，可能返回一个 Document，但其 text 为空
        for doc in documents:
            assert doc.text.strip() == "", "文档内容为空时，text 应为空"
        
    # Edge Case 3：查询为空
    from src.retriever.hybrid_retriever import HybridRetriever
    # 假设已有 vector_retriever 和 bm25_retriever（可用空列表初始化）
    class DummyRetriever:
        def retrieve(self, query, top_k=5):
            return []
    dummy_vector = DummyRetriever()
    dummy_bm25 = DummyRetriever()
    hybrid_retriever = HybridRetriever(dummy_vector, dummy_bm25)
    results = hybrid_retriever.retrieve("", top_k=3)
    assert results == [], "空查询应返回空结果"
    
    # Edge Case 4：BM25 得分均为 0 的情况
    from src.data_loader.markdown_loader import Document
    docs = [Document("全是无关内容。", {}), Document("没有包含关键词。", {})]

    from src.retriever.bm25_retriever import BM25Retriever
    bm25_retriever = BM25Retriever(docs)
    results = bm25_retriever.retrieve("不存在的查询词", top_k=2)
    assert results == [], "查询词在所有文档中均未出现时，应返回空列表"
    


def testing():
    print("==== 运行正常流程测试 ====")
    test_normal_flow()
    print("==== 正常流程测试完成 ====")
    print("==== 运行边缘情况测试 ====")
    test_edge_cases()
    print("==== 边缘情况测试完成 ====")

if __name__ == "__main__":
    testing()