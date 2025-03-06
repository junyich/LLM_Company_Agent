from pathlib import Path
from typing import List, Dict

# 简单定义一个 Document 类，用于存储文本及元数据
class Document:
    def __init__(self, text: str, metadata: Dict):
        self.text = text
        self.metadata = metadata

    def __repr__(self):
        return f"Document(text={self.text[:30]}..., metadata={self.metadata})"


class MarkdownLoader:
    """
    MarkdownLoader 用于加载指定文件夹内的 Markdown 文件及其相关图片。
    
    规则：
    - 扫描文件夹内所有 .md 文件。
    - 对于每个 .md 文件，检查是否存在同名的图片文件夹（例如 Doc1.md 对应 Doc1_images）。
    - 将图片文件路径以列表形式存入文档的 metadata 中。
    """
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def load_data(self) -> List[Document]:
        documents = []
        base_path = Path(self.folder_path)
        
        # 遍历文件夹内所有 .md 文件
        for md_file in base_path.glob("*.md"):
            try:
                with md_file.open("r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading {md_file}: {e}")
                continue

            # 检查是否存在对应的图片文件夹
            images_folder = base_path / f"{md_file.stem}_images"
            images = []
            if images_folder.exists() and images_folder.is_dir():
                # 仅获取常见图片格式
                for image_file in images_folder.iterdir():
                    if image_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif"]:
                        images.append(str(image_file.resolve()))

            metadata = {
                "file_name": md_file.name,
                "file_path": str(md_file.resolve()),
                "images": images
            }
            documents.append(Document(text=content, metadata=metadata))
        
        return documents


# 如果需要进行简单测试，可以取消下面的注释
if __name__ == "__main__":
    loader = MarkdownLoader(folder_path=r"C:\Users\12439\Downloads\亲亲我的大钢影\data_cleaning\markdown_data")
    docs = loader.load_data()
    for doc in docs:
        print(doc)
