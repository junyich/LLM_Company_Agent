import os
from src.data_loader import MarkdownLoader, Document
from src.preprocessor import HierarchicalTextSplitter
from src.embedding import Embedder
from src.indexer import VectorIndex
from src.retriever import VectorRetriever
from src.retriever import BM25Retriever
from src.retriever import HybridRetriever
from src.llm_interface import OnlineLLM
from src.configs.config import STORAGE_DIR


class AskQuestion:
    """
    AskQuestion 类用于从指定的 Markdown 文件夹加载文本，
    然后基于文本内容生成五个刁钻且具有挑战性的问题。
    """
    def __init__(self, markdown_folder: str):
        """
        :param markdown_folder: 存放 Markdown 文件的文件夹路径
        :param api_key: OpenAI API Key
        :param model_name: 使用的 GPT 模型名称，默认为 "gpt-4o"
        """
        self.markdown_folder = markdown_folder

    def load_markdown_text(self) -> str:
        """
        使用 MarkdownLoader 加载指定文件夹下所有 Markdown 文件，
        并将所有文档的文本内容合并为一个长文本返回。
        """
        loader = MarkdownLoader(self.markdown_folder)
        documents = loader.load_data()
        combined_text = "\n".join([doc.text for doc in documents])
        return combined_text

    def generate_questions(self) -> str:
        """
        基于加载的 Markdown 文档内容构造 prompt，
        调用 OnlineLLM 模型生成五个基于文本内容且刁钻的问题，
        要求每个问题独占一行，并返回生成的文本。
        """
        # 加载所有 Markdown 文本内容
        md_text = self.load_markdown_text()
        # 构造 prompt
        prompt = (
            "请根据以下Markdown文档内容生成10个问题，要求每个问题必须基于文本内容，且尽量刁钻、具有挑战性：\n\n"
            "【Markdown 文档内容】:\n"
            f"{md_text}\n\n"
            "请只返回问题列表，每个问题独占一行，不需要额外解释。"
        )
        # 调用 OnlineLLM 模型
        self.llm = OnlineLLM(temperature=0.5)
        response = self.llm.complete(prompt)
        return response.text
    
    def evaluate(self, answer_text: str) -> str:
        """
        根据 Markdown 文档内容和 RAG 系统生成的回答构造 prompt，
        调用 GPT-4o 模型进行评分，并返回评分结果及详细评价说明。

        不会对对话记录产生影响，可以直接反复call这个function
        
        :param answre_text: RAG 系统生成的回答文本
        :return: 模型返回的评分和评价说明
        """
        prompt = (
            "请根据以下 RAG 系统生成的回答进行评分。\n\n"
            "【RAG 系统回答】:\n"
            f"{answer_text}\n\n"
            "请根据回答是否符合文本内容的事实，严格评分，给出一列的1到10的评分（10分为最佳）对每个问题输出1个精准到第一个小数点的评分，不需要给出理由，用回车分割，严格一点"
            "比如：\n3.2\n7.6\n"
        )
        # 获取模型返回的 LLMResponse 对象，并提取其中的文本
        mesg = self.llm.messages
        evaluation = self.llm.complete(prompt)
        score_lst = evaluation.text.split("\n")
        self.llm.messages = mesg
        return list(map(lambda x: float(x), score_lst))

def initialize_system(target_dir):
    # 1. 加载 Markdown 文档（从 config 中指定的目录读取）
    loader = MarkdownLoader(target_dir)
    documents = loader.load_data()
    print(f"Loaded {len(documents)} Markdown documents from {target_dir}.")

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
    llm = OnlineLLM(model_name="gpt-4o-mini")

    return hybrid_retriever, llm

def ask_10_questions(target_dir):

    # -------------------------------
    # 1. 生成问题：调用 AskQuestion 类生成五个刁钻的问题，并按行拆分
    # -------------------------------
    hybrid_retriever, llm = initialize_system(target_dir=target_dir)
    ask_q = AskQuestion(markdown_folder=target_dir)
    questions_str = ask_q.generate_questions()
    # 按行拆分为单独问题，并过滤掉空行
    questions_list = [q.strip() for q in questions_str.splitlines() if q.strip()]
    
    print("生成的问题：")
    for idx, question in enumerate(questions_list, 1):
        print(f"问题{idx}: {question}")
    print("=" * 60)
    
    # -------------------------------
    # 2. 对每个问题分别处理：检索、生成回答和评价
    # -------------------------------
    responses = ""
    raw_response = ""

    for idx, question in enumerate(questions_list, 1):
        
        # 利用混合检索器获取与该问题相关的文档
        retrieved_docs = hybrid_retriever.retrieve(question, top_k=7)
        if not retrieved_docs:
            print("  未检索到相关文档。")
            continue

        # 构造 prompt，将检索到的内容整合为上下文，并附上当前问题
        combined_context = "\n".join([doc.text for doc in retrieved_docs])
        prompt = f"基于以下内容回答问题：\n{combined_context}\n问题：{question}\n回答："
        responses = f"{responses}\n{(llm.complete(prompt).text)}"

    # TODO: raw responses
    llm.clear_chat()
    raw_prompt = f"基于以下内容回答问题：\n问题：{questions_list}\n回答："
    raw_response = f"{raw_response}\n{(llm.complete(raw_prompt).text)}"
    
    scores = ask_q.evaluate(responses)
    scores2 = ask_q.evaluate(raw_response)
    print("scores: ")
    print(scores)
    print("sccores(with no rag): ")
    print(scores2)
    avg_score1 = sum(scores) / len(scores) if scores else 0  # 避免除零错误
    avg_score2 = sum(scores2) / len(scores2) if scores2 else 0  # 避免除零错误
    return [avg_score1, avg_score2] 

def main():
    processed_dir = "./testing/eval_samples_ready_to_use"
    total_scores_1 = 0  # RAG 方式得分
    total_scores_2 = 0  # 直接 LLM 方式得分
    dir_lst = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    for dir in dir_lst:
        avg_scores = ask_10_questions(dir)  # 现在返回的是 [RAG平均分, 直接LLM平均分]
        total_scores_1 += avg_scores[0]
        total_scores_2 += avg_scores[1]
    n = len(dir_lst)
    print(f"共测试 {n * 10} 个问题")
    print(f"综合得分（RAG 方式）：{total_scores_1 / n}")
    print(f"综合得分（直接 LLM 方式）：{total_scores_2 / n}")
    print(f"最终得分列表: {[total_scores_1 / n, total_scores_2 / n]}")  # 以列表形式返回

if __name__ == "__main__":
    main()
    # 给我说说我们的设备验收方案