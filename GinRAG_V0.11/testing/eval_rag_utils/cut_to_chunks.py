import os

def split_file_into_chunks(input_file, output_folder, max_chunks):
    """
    把input_file按500行切分，每份存到output_folder里的子文件夹，最多存max_chunks份。
    """
    # 读取所有行
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_lines = len(lines)
    # 每份500行，最多分成多少份
    chunk_size = 500
    total_chunks = (total_lines + chunk_size - 1) // chunk_size  # 向上取整

    # 如果总份数大于最大允许份数，就只保留前max_chunks份
    if total_chunks > max_chunks:
        total_chunks = max_chunks

    # 创建存放分割文件的主文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 分割并保存每份
    for i in range(total_chunks):
        chunk_lines = lines[i * chunk_size:(i + 1) * chunk_size]
        
        # 每份放到一个独立子文件夹
        sub_folder = os.path.join(output_folder, f"chunk{i + 1}")
        os.makedirs(sub_folder, exist_ok=True)

        # 保存为每份的文件名
        chunk_file = os.path.join(sub_folder, f"text_part_{i + 1}.md")

        with open(chunk_file, "w", encoding="utf-8") as f:
            f.writelines(chunk_lines)

        print(f"保存第{i+1}份到：{chunk_file}")

    print(f"处理完成，总共生成{total_chunks}份（每份500行，最多{max_chunks}份）。")

if __name__ == "__main__":
    input_md_file = r"F:\Project\备份文件\input.txt"
    output_base_folder = r"F:\Project\GinRAG_V0.11\test\eval_samples_ready_to_use"  # 存放分割文件的主目录
    max_chunks = 10  # 最多分割成10份

    split_file_into_chunks(input_md_file, output_base_folder, max_chunks)
