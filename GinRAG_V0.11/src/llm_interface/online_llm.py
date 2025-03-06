from openai import OpenAI
from src.configs.config import API_DIR

# 定义 LLMResponse 类，用于封装模型返回的文本
class LLMResponse:
    """
    用于封装大模型生成结果的简单响应对象。
    """
    def __init__(self, text: str):
        self.text = text

class OnlineLLM:
    """
    OnlineLLM 利用 OpenAI API 调用 GPT 模型，
    提供 complete(prompt) 方法进行对话，
    并通过 clear_chat() 方法清空聊天历史，
    实现连续对话和重置对话的功能。
    """
    def __init__(self, model_name: str = "gpt-4o-mini-2024-07-18", api_key: str = None, temperature=0.1, max_tokens=1024):
        self.model_name = model_name
        if api_key is None:
            with open(API_DIR, 'r', encoding='utf-8') as file:
                api_key = file.read().strip()
        self.client = OpenAI(api_key=api_key)
        # 初始化对话历史，预设 system 消息
        self.messages = []
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> LLMResponse:
        # 添加用户消息到历史记录中
        self.messages.append({"role": "user", "content": prompt})
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                # max_completion_tokens=self.max_tokens
            )
            # 使用 .content 获取消息内容
            output = response.choices[0].message.content.strip()
            # 将模型回复添加到历史记录中
            self.messages.append({"role": "assistant", "content": output})
            return LLMResponse(text=output)
        except Exception as e:
            error_msg = f"Error calling GPT model: {str(e)}"
            print(error_msg)
            return LLMResponse(text=error_msg)

    def clear_chat(self):
        """清空聊天历史，只保留 system 消息。"""
        self.messages = []

# 测试代码
if __name__ == "__main__":
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

    # 清空聊天记录
    gpt_llm.clear_chat()
    prompt3 = "请介绍一下自然语言处理的发展历程。"
    response3 = gpt_llm.complete(prompt3)
    print("\nGPT LLM Response 3 (after clearing chat):")
    print(response3.text)

