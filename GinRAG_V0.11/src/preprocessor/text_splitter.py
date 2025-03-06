# -*- coding: utf-8 -*-

from typing import List
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import Document

class HierarchicalTextSplitter:
    """
    HierarchicalTextSplitter 利用层次化分段技术对文档进行切分处理，
    支持多层次的 chunk 大小设置，确保既能保留全局结构也能提取细粒度信息。

    参数:
        chunk_sizes: List[int]，例如 [2048, 512, 256]，从大到小的分段尺寸，
                     较大的尺寸用于整体框架，较小的尺寸用于细粒度补充。
        chunk_overlap: int，每一层分段时的重叠长度，需小于所有层级的 chunk_size。
    """
    def __init__(self, chunk_sizes: List[int] = [2048, 512, 256], chunk_overlap: int = 50):
        # 检查 overlap 是否符合要求：必须小于所有 chunk_size
        if any(chunk_overlap >= cs for cs in chunk_sizes):
            raise ValueError("chunk_overlap must be smaller than all values in chunk_sizes")
        self.chunk_sizes = chunk_sizes
        self.chunk_overlap = chunk_overlap
        # 显式传递 chunk_overlap 参数
        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=chunk_sizes,
            chunk_overlap=chunk_overlap
        )

    def split_text(self, text: str) -> List[str]:
        """
        将单个文本字符串进行层次化分段处理。

        参数:
            text: 原始文本字符串。

        返回:
            List[str]，分段后的文本块（叶子节点）。
        """
        document = Document(text=text)
        # 获取层次化分段节点
        nodes = self.node_parser.get_nodes_from_documents([document])
        # 提取所有叶子节点，叶子节点通常对应最细粒度的分段
        leaf_nodes = get_leaf_nodes(nodes)
        # 返回每个叶子节点的文本内容
        return [node.text for node in leaf_nodes]

    def split_documents(self, docs: List[Document]) -> List[str]:
        """
        针对一批文档进行层次化分段处理。

        参数:
            docs: List[Document]，每个文档包含 text 字段。

        返回:
            List[str]，所有文档分段后的文本块合集。
        """
        nodes = self.node_parser.get_nodes_from_documents(docs)
        leaf_nodes = get_leaf_nodes(nodes)
        return [node.text for node in leaf_nodes]


# 测试代码
if __name__ == "__main__":
    sample_text = (
        "第一段：机器学习是人工智能的重要分支。它通过从数据中提取规律，帮助计算机做出预测或决策。"
        "在过去几年中，机器学习已经在图像识别、语音处理和自然语言处理等领域取得了突破性进展。\n\n"
        "第二段：深度学习是一种特殊的机器学习方法，其核心在于使用多层神经网络。"
        "这种方法能够自动抽取数据中的高层次特征，大幅度提高模型性能。"
        "不过，深度学习对数据量和计算资源的要求也比较高。\n\n"
        "第三段：未来的发展趋势是结合多种技术进行混合建模。"
        "例如，将传统机器学习方法与深度学习模型相结合，可以在保持模型解释性的同时提高预测精度。"
    )

    # 注意：确保 chunk_overlap 小于所有 chunk_size，这里设置为 0
    splitter = HierarchicalTextSplitter()
    chunks = splitter.split_text(sample_text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}")
