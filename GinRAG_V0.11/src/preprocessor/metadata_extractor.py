import yaml
import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

class MetadataExtractor:
    """
    MetadataExtractor 用于从 Markdown 文件中提取元数据，
    包括内嵌的 YAML Front Matter 和文件系统信息。
    """
    
    def __init__(self):
        pass

    def extract_from_file(self, file_path: str) -> Dict:
        """
        从给定的 Markdown 文件中提取元数据。
        
        提取内容包括：
         - YAML Front Matter 中的元数据（如 title、author、date、tags 等）
         - 文件名、文件路径、文件修改时间等文件系统信息
        
        参数:
            file_path: 文件的绝对或相对路径
        
        返回:
            metadata: Dict，包含提取的元数据
        """
        file_path_obj = Path(file_path)
        metadata = {
            "file_name": file_path_obj.name,
            "file_path": str(file_path_obj.resolve()),
            "modified_time": self._get_modified_time(file_path_obj)
        }
        
        try:
            with file_path_obj.open("r", encoding="utf-8") as f:
                content = f.read()
            front_matter, _ = self._parse_yaml_front_matter(content)
            if front_matter:
                # 将 front matter 中的内容合并到 metadata 中
                metadata.update(front_matter)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        return metadata

    def _parse_yaml_front_matter(self, text: str) -> Tuple[Optional[Dict], str]:
        """
        解析 Markdown 文本中的 YAML Front Matter。
        
        参数:
            text: Markdown 文件的完整文本
        
        返回:
            front_matter: 如果存在则为字典，否则为 None
            content: 去除 front matter 后的正文内容
        """
        if text.startswith("---"):
            # 尝试查找第二个 '---'
            parts = text.split("---", 2)
            if len(parts) >= 3:
                try:
                    front_matter = yaml.safe_load(parts[1])
                except Exception as e:
                    print(f"Error parsing YAML front matter: {e}")
                    front_matter = None
                content = parts[2].strip()
                return front_matter, content
        return None, text

    def _get_modified_time(self, file_path_obj: Path) -> str:
        """
        获取文件的最后修改时间，并格式化为 ISO 格式字符串。
        """
        try:
            mtime = file_path_obj.stat().st_mtime
            dt = datetime.datetime.fromtimestamp(mtime)
            return dt.isoformat()
        except Exception as e:
            print(f"Error getting modified time for {file_path_obj}: {e}")
            return ""

# 简单测试
if __name__ == "__main__":
    test_file = r"C:\Users\12439\Downloads\亲亲我的大钢影\data_cleaning\markdown_data\商务部分子集_产品技术培训.md"  # 请替换为实际测试文件路径
    extractor = MetadataExtractor()
    metadata = extractor.extract_from_file(test_file)
    print("Extracted Metadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")
