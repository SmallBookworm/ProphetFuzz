import os
import datetime
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

utils_dir = os.path.dirname(os.path.abspath(__file__))
prompt_dir = os.path.abspath(os.path.join(utils_dir, ".."))

class GPTUtils:
    def __init__(self) -> None:
        config_dir = os.path.join(prompt_dir, "config")
        # 加载环境变量
        if os.path.exists(os.path.join(config_dir, ".env")):
            load_dotenv(os.path.join(config_dir, ".env"))
        
        # 1. 初始化 OpenAI 客户端
        # 注意：这里使用了兼容模式的 Base URL
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" # 注意：去掉了末尾的 endpoint 路径
        )
        pass

    # 2. 使用装饰器实现重试逻辑 (替代原来的 while True 循环)
    @retry(
        retry=retry_if_exception_type((APIConnectionError, RateLimitError)), # 指定重试的异常类型
        stop=stop_after_attempt(3), # 最多重试3次
        wait=wait_exponential(multiplier=1, max=10) # 指数退避等待
    )
    def queryOpenAI(self, prompt, model, temperature, n) -> list:
        print(f"[INFO] Querying OpenAI API with prompt: {prompt[:50]}...") # 打印提示信息，截断过长的提示
        print(f"[INFO] Model: deepseek-v3.2, Temperature: {temperature}, N: {n}")
        try:
            # 3. 使用 SDK 的 chat.completions.create 方法
            response = self.client.chat.completions.create(
                model="deepseek-v3.2", # 模型名称
                messages=[
                    {
                        "role": "system", 
                        "content": f"You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.\nKnowledge cutoff: 2023-04\nCurrent date: {datetime.date.today()}"
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    },
                ],
                # stream=False, # SDK 默认就是非流式，通常不需要显式声明
                temperature=temperature,
                n=n
            )

            ret_messages = []
            for choice in response.choices:
                message = choice.message.content
                # 检查是否因为长度限制而停止
                if choice.finish_reason == 'length':
                    print("[INFO] The maximum number of tokens specified in the request was reached.")
                ret_messages.append(message)

            return ret_messages
            
        except Exception as e:
            print(f"[x] Failed to call OpenAI API: {e}")
            # 如果你想让程序在最终失败时退出，可以在这里处理
            # exit(1) # 根据你的需求决定是否保留
            raise # 重新抛出异常以便 retry 装饰器捕获