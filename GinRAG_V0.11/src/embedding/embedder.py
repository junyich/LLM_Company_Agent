from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    """
    Embedder 利用预训练的 SentenceTransformer 模型对文本进行嵌入，
    支持批量处理，并返回嵌入向量。
    
    参数:
        model_name: 使用的嵌入模型名称，默认采用 "all-MiniLM-L6-v2"，
                    可根据需要替换为其他本地部署的模型。
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """
        对输入文本列表进行批量嵌入，返回嵌入向量数组。

        参数:
            texts: List[str]，文本列表
            batch_size: 每个批次处理的文本数量

        返回:
            np.ndarray，形状为 (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return np.array(embeddings)

# 测试代码
if __name__ == "__main__":
    embedder = Embedder()
    sample_texts = [
        "机器学习是人工智能的重要分支。",
        "深度学习在图像识别中有广泛应用。"
    ]
    embeddings = embedder.embed_text(sample_texts)
    print("Embeddings shape:", embeddings.shape)
