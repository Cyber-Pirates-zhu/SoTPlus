import ast
import json
from typing import List, Dict

import openai
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from SoT_plus.fundation_models.GPT import Agent
from SoT_plus.parser import Parser
from SoT_plus.prompter import Prompter
from sample import s_Prompter, s_Parser

openai.api_key = "api key"
openai.api_base = "https://api.openai.com/v1"


def Logic_dependency_detection(lm: str, prompter: Prompter, parser: Parser, known_condition: List[str],
                               additional_known_conditions: List[str], question: str) -> Dict:
    prompt = prompter.Logic_dependency_detection_prompt(known_information=known_condition,
                                                        additional_known_information=additional_known_conditions,
                                                        question=question)
    agent = Agent(model=lm)
    responses, conversationHistory = agent.conversation(prompt)
    result = parser.Logic_dependency_detection_answer(responses)
    return result


if __name__ == "__main__":
    handler = Agent(model="gpt-3.5-turbo")
    lm = "gpt-3.5-turbo"
    prompter = s_Prompter()
    parser = s_Parser()
    Questions = ['What was the total amount of milk Mrs.Lim got yesterday?', 'What was the total amount of milk Mrs.Lim got yesterday?', 'How much milk did Mrs.Lim get this morning?']
    results = [{}] * len(Questions)  # 创建与输入大小相同的结果占位列表
    known_condition = ['Mrs.Lim milks her cows twice a day.', 'Yesterday morning, she got 68 gallons of milk.', 'Yesterday evening, she got 82 gallons of milk.', 'This morning, she got 18 gallons fewer than she had yesterday morning.', 'After selling some gallons of milk in the afternoon, Mrs.Lim has only 24 gallons left.']
    additional_known_conditions = []
    with ThreadPoolExecutor() as logic_detection_executor:
        # 使用enumerate跟踪消息的索引
        futures = {
            logic_detection_executor.submit(Logic_dependency_detection, lm, prompter, parser, known_condition, additional_known_conditions, question): idx
            for idx, question in enumerate(Questions)
        }
        for future in as_completed(futures):
            idx = futures[future]  # 获取任务对应的输入索引
            result = future.result()  # 解包结果
            results[idx] = result  # 按索引存储结果

    print("无关性检测:", results)
