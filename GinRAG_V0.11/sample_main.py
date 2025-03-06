import os
from src.data_loader import MarkdownLoader, Document
from src.preprocessor import HierarchicalTextSplitter
from src.embedding import Embedder
from src.indexer import VectorIndex
from src.retriever import VectorRetriever
from src.retriever import BM25Retriever
from src.retriever import HybridRetriever
from src.llm_interface import LocalLLM
from src.configs.config import STORAGE_DIR, MARKDOWN_DIR


def initialize_system():
    # 1. 加载 Markdown 文档（从 config 中指定的目录读取）
    loader = MarkdownLoader(MARKDOWN_DIR)
    documents = loader.load_data()
    print(f"Loaded {len(documents)} Markdown documents from {MARKDOWN_DIR}.")

    # 2. 文本预处理与分段
    splitter = HierarchicalTextSplitter(chunk_sizes=[2048, 512, 256], chunk_overlap=50)
    processed_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc.text)
        for chunk in chunks:
            processed_docs.append(Document(text=chunk, metadata=doc.metadata))
    print(f"After splitting, obtained {len(processed_docs)} text chunks.")

    # 3. 嵌入与向量索引构建
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    vector_dim = 384  # 请确保与模型输出维度一致
    index_path = os.path.join(STORAGE_DIR, "faiss.index")
    vector_index = VectorIndex(embedder=embedder, vector_dim=vector_dim, index_path=index_path)
    vector_index.index_documents(processed_docs)
    print(f"Indexed {vector_index.index.ntotal} text chunks into FAISS.")

    # 4. 构建检索器
    vector_retriever = VectorRetriever(vector_index)
    bm25_retriever = BM25Retriever(processed_docs)
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

    # 5. 初始化本地 LLM
    llm = LocalLLM(model_name="deepseek-r1")

    return hybrid_retriever, llm

def main():
    hybrid_retriever, llm = initialize_system()

    print("欢迎使用文档问答系统，输入 '/end' 结束对话。")
    while True:
        query = input("请输入您的问题：").strip()
        if query == "/end":
            print("对话结束，再见！")
            break
        if not query:
            print("输入为空，请重新输入。")
            continue

        # 利用混合检索器获取相关文档
        retrieved_docs = hybrid_retriever.retrieve(query, top_k=7)
        if not retrieved_docs:
            print("未检索到相关文档，请尝试其他问题。")
            continue

        # 显示检索到的文档摘要（可选）
        print("检索结果：")
        for doc in retrieved_docs:
            print(f"- {doc.text}") 
        print("-" * 40)

        # 构造 prompt，将检索到的内容整合为上下文
        combined_context = "\n".join([doc.text for doc in retrieved_docs])
        prompt = f"基于以下内容回答问题：\n{combined_context}\n问题：{query}\n回答："
        response = llm.complete(prompt)
        print("生成回答：")
        print(response.text)
        print("=" * 60)

if __name__ == "__main__":
    main()
    # 给我说说我们的设备验收方案