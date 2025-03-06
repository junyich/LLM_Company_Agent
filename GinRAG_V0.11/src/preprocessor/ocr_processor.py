import os
import easyocr
from PIL import Image, UnidentifiedImageError

class OCRProcessor:
    """
    OCRProcessor 利用 EasyOCR 对图片进行文本识别。

    依赖：
      - easyocr：支持多语言的 OCR 识别库
    """
    
    def __init__(self, languages: list = ['en', 'ch_sim']):
        """
        初始化 OCRProcessor。

        参数:
            languages: 列表，指定 OCR 使用的语言（例如 ['en'] 或 ['ch_sim','en']）
        """
        self.reader = easyocr.Reader(languages, gpu=False)  # gpu 参数根据是否有 GPU 可用进行设置

    def process_image(self, image_path: str) -> str:
        """
        对单张图片进行 OCR 识别，返回识别后的文本。

        参数:
            image_path: 图片文件的路径

        返回:
            识别后的文本内容。
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # 尝试使用 Pillow 打开图片
            with Image.open(image_path) as img:
                # 先转换为 RGBA，处理带有透明度的调色板图像
                img = img.convert("RGBA")
                # 将透明区域合成到白色背景上
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # 使用 alpha 通道作为 mask
                processed_img = background

            # 转换为 numpy 数组传给 EasyOCR
            import numpy as np
            img_np = np.array(processed_img)
            
            # 使用 EasyOCR 识别，detail=0 返回纯文本列表
            result = self.reader.readtext(img_np, detail=0)
            # 将识别的文本片段合并为一段
            text = " ".join(result)
            return text.strip()
        except UnidentifiedImageError as e:
            print(f"UnidentifiedImageError processing image {image_path}: {e}")
            return ""
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return ""
    
    def process_images(self, image_paths: list) -> dict:
        """
        批量处理多张图片，返回每张图片的 OCR 结果。

        参数:
            image_paths: 图片路径列表

        返回:
            字典，键为图片路径，值为识别后的文本。
        """
        results = {}
        for path in image_paths:
            results[path] = self.process_image(path)
        return results


# 测试代码
if __name__ == "__main__":
    # 示例图片路径，请根据实际情况替换
    test_image = r"C:\Users\12439\Downloads\亲亲我的大钢影\data_cleaning\markdown_data\资格部分子集_公司组织机构图_images\image_inline_1.png"
    ocr_processor = OCRProcessor()
    result_text = ocr_processor.process_image(test_image)
    print("OCR Result:")
    print(result_text)
