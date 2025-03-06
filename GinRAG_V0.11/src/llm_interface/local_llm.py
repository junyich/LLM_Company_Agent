import subprocess

class LLMResponse:
    """
    用于封装大模型生成结果的简单响应对象。
    """
    def __init__(self, text: str):
        self.text = text

class LocalLLM:
    """
    LocalLLM 利用本地安装的 Ollama 调用 deepseek‑r1 模型，
    提供 complete(prompt) 接口将用户的 prompt 传递给模型，
    并返回生成的文本结果。

    通用处理方式：
      - 统一接口：实现 complete(prompt) 方法返回 LLMResponse 对象。
      - 输入预处理：可在此处扩展对 prompt 的格式转换。

    特殊处理方式：
      - 使用 subprocess 调用本地 Ollama 命令行接口，
        这里不使用 flag，而是将 prompt 作为标准输入传递给命令。
      - 指定 encoding="utf-8" 以确保中文字符能正确编码。
      - 捕获命令执行中的错误，并返回相应的错误信息。
    """
    def __init__(self, model_name: str = "deepseek-r1"):
        self.model_name = model_name

    def complete(self, prompt: str) -> LLMResponse:
        """
        调用本地 Ollama 模型 deepseek‑r1 生成文本。

        参数:
            prompt: 输入的提示文本

        返回:
            LLMResponse 对象，包含生成的文本。
        """
        try:
            # 构造命令行参数，不使用 flag，而是直接传入模型名称
            command = ["ollama", "run", self.model_name]
            # 使用 subprocess.run 将 prompt 作为标准输入传递给命令，
            # 并强制指定 encoding 为 "utf-8"
            result = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True
            )
            output = result.stdout.strip()
            return LLMResponse(text=output)
        except subprocess.CalledProcessError as e:
            error_msg = f"Error calling local LLM: {e.stderr}"
            print(error_msg)
            return LLMResponse(text=error_msg)

# 测试代码
if __name__ == "__main__":
    llm = LocalLLM(model_name="deepseek-r1")
    prompt = "请总结一下当前深度学习在图像识别领域的最新进展。"
    response = llm.complete(prompt)
    print("Local LLM Response:")
    print(response.text)
