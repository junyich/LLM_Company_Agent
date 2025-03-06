from testing.eval_rag_utils import ask_10_questions, split_file_into_chunks, extract_mediawiki_text
from testing.src_tests import testing, test_vector, bm25_test, testing_hybrid
import os
import sys
import io

EVAL_DATA = r"F:\Project\wiki测试\wikitest.xml-p1p41242"
EVAL_CHUNKS = 10

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_src_tests():
    test_vector()
    bm25_test()
    testing_hybrid()
    testing()

def run_eval():
    target_dir = f"{os.path.dirname(EVAL_DATA)}\\input.txt"
    extract_mediawiki_text(EVAL_DATA, target_dir)
    processed_dir = "./testing/eval_samples_ready_to_use"
    split_file_into_chunks(target_dir, processed_dir, 10)

    total_scores_1 = 0  # RAG 评分总和
    total_scores_2 = 0  # 直接 LLM 评分总和
    dir_lst = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]

    for dir in dir_lst:
        avg_scores = ask_10_questions(os.path.join(processed_dir, dir))  # 返回 [RAG 评分, 直接 LLM 评分]
        total_scores_1 += avg_scores[0]
        total_scores_2 += avg_scores[1]

    n = len(dir_lst)
    print(f"共测试 {n*10} 个问题")
    print(f"综合得分（RAG）：{total_scores_1 / n}")
    print(f"综合得分（直接 LLM）：{total_scores_2 / n}")
    print(f"最终得分列表: {[total_scores_1 / n, total_scores_2 / n]}")  # 以列表形式输出

if __name__ == "__main__":
    # run_src_tests()
    # TODO: debug一下
    run_eval()
    