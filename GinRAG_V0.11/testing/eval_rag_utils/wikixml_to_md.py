import xml.etree.ElementTree as ET

def extract_mediawiki_text(xml_file, output_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    texts = []
    # 用 {*}page 来匹配任何命名空间下的 <page>
    for page in root.findall(".//{*}page"):
        # 在 <page> 下找 <revision> 再找 <text>
        text_elem = page.find(".//{*}text")
        if text_elem is not None and text_elem.text:
            texts.append(text_elem.text.strip())

    with open(output_file, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n\n==========\n\n")

if __name__ == "__main__":
    extract_mediawiki_text(
        r"F:\Project\wiki测试\wikitest.xml-p1p41242",
        r"F:\Project\备份文件\input.txt"
        )
