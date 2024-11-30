import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import numpy as np

# 设置 OpenAI API 密钥
openai.api_key = "api key"
openai.api_base = "https://api.openai.com/v1"
class Agent:
    def __init__(self, Conversation_History: List[Dict] = None, model="gpt-4"):
        if Conversation_History is not None:
            self.history = Conversation_History
        else:
            self.history = []
        self.model = model

    def chat(self, messages, temperature: int = 0):

        system_message = """
        You are a helpful assistant. Always follow the instructions precisely and output the response exactly in the requested format.
        """

        # 将system_message插入到对话的开头
        Input = messages.copy()
        Input.insert(0, {"role": "system", "content": system_message})
        # 向模型发送对话并返回模型生成的对话
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=Input,
            temperature=temperature,
            max_tokens=2000
        )
        """
        response.choices[0]: 一个列表, 存储着所有LLM的候选答案
        每个答案是一个字典, 会有一个"message"键
        message中存放着一个字典, 字典的"content"键存着LLM生成的答案
        """
        reply = response["choices"][0]["message"]["content"]
        return reply

    def conversation(self, message, SessionWindowLength=0, num_response: int = 1):

        user_input = message
        # 根据访问窗口长度, 将总历史中从后往前数等于窗口长度的内容提取出来作为prompt
        conversationHistory = []
        if len(self.history) <= SessionWindowLength:
            conversationHistory = self.history.copy()
        else:
            begin = len(self.history) - SessionWindowLength
            for i in range(begin, len(self.history)):
                conversationHistory.append(self.history[i])
        # 将最新的用户问题添加到query窗口中, 一起作为prompt
        conversationHistory.append({"role": "user", "content": user_input})
        self.history.append({"role": "user", "content": user_input})

        Temperature = np.linspace(0, 0.9, num_response)
        # 生成答案
        Response = []
        for temperature in Temperature:
            response = self.chat(conversationHistory, temperature)
            Response.append(response)

        # 将生成的答案加入query窗口中
        conversationHistory.append({"role": "assistant", "content": Response})
        self.history.append({"role": "assistant", "content": Response})

        return Response, conversationHistory

    def parallel_conversations(self, messages: List[str], num_response: int = 1,
                               session_window_lengths: List[int] = None) \
            -> List:
        if session_window_lengths is None:
            session_window_lengths = [0] * len(messages)

        results = [None] * len(messages)  # 创建与输入大小相同的结果占位列表
        with ThreadPoolExecutor() as executor:
            # 使用enumerate跟踪消息的索引
            futures = {
                executor.submit(self.conversation, message, length, num_response): idx
                for idx, (message, length) in enumerate(zip(messages, session_window_lengths))
            }
            for future in as_completed(futures):
                idx = futures[future]  # 获取任务对应的输入索引
                result, _ = future.result()  # 解包结果
                results[idx] = result  # 按索引存储结果
        return results


if __name__ == "__main__":
    '''
        GPT 4
            gpt-4
        GPT 3.5
            gpt-3.5-turbo 
            gpt-3.5-turbo-16k 
    '''
    handler = Agent(model="gpt-3.5-turbo")
    m = ["What is the result of 1+1?", "What is the result of 1+2?", "What is the result of 1+3?"]
    Result = handler.parallel_conversations(m, num_response=3)
    print(Result)
