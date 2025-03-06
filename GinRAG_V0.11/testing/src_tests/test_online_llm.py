from src.llm_interface import OnlineLLM

gpt_llm = OnlineLLM()

# 第一次对话
prompt1 = "请总结一下当前深度学习在图像识别领域的最新进展。"
response1 = gpt_llm.complete(prompt1)
print("GPT LLM Response 1:")
print(response1.text)

# 继续对话
prompt2 = "能再详细一点吗？"
response2 = gpt_llm.complete(prompt2)
print("\nGPT LLM Response 2:")
print(response2.text)

# 清空聊天
gpt_llm.clear_chat()
prompt3 = "请介绍一下自然语言处理的发展历程。"
response3 = gpt_llm.complete(prompt3)
print("\nGPT LLM Response 3 (after clearing chat):")
print(response3.text)