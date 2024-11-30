from __future__ import annotations

import itertools
import logging
import string
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from typing import Iterator, Optional, List, Dict

# from SoT_plus.fundation_models.Llama import Agent
from SoT_plus.fundation_models.GPT import Agent
from SoT_plus.method_of_SoTPlus.middle_goal import MiddleGoal
from SoT_plus.parser import Parser
from SoT_plus.prompter import Prompter


def Logic_dependency_detection(lm: str, prompter: Prompter, parser: Parser, known_condition: List[str],
                               additional_known_conditions: List[str], question: str) -> Dict:
    prompt = prompter.Logic_dependency_detection_prompt(known_information=known_condition,
                                                        additional_known_information=additional_known_conditions,
                                                        question=question)
    agent = Agent(model=lm)
    responses, conversationHistory = agent.conversation(prompt)
    result = parser.Logic_dependency_detection_answer(responses)
    return result


def Direct_logical_duplicate_detection(lm: str, prompter: Prompter, parser: Parser, known_condition: List[str],
                                       additional_known_conditions: List[str], goal: str, question: str) -> Dict:
    prompt = prompter.Create_known_information([goal])
    agent = Agent(model=lm)
    responses, conversationHistory = agent.conversation(prompt)
    result = parser.parse_create_known_information_answer(responses)
    result = result[0]["known information"]
    Know_information = additional_known_conditions.copy()
    Know_information.append(result)

    prompt = prompter.Direct_logical_duplicate_detection_prompt(known_condition, Know_information, question)
    responses, conversationHistory = agent.conversation(prompt)
    result = parser.Direct_logical_duplicate_detection_answer(responses)
    return result


def Indirect_logical_duplication_detection(lm: str, prompter: Prompter, parser: Parser, known_condition: List[str],
                                           additional_known_conditions: List[str], goal: str, information: str) -> Dict:
    prompt = prompter.Indirect_logical_duplication_detection_prompt(known_condition,
                                                                    additional_known_conditions,
                                                                    information,
                                                                    goal)
    agent = Agent(model=lm)
    responses, conversationHistory = agent.conversation(prompt)
    result = parser.Indirect_logical_duplication_detection_answer(responses)
    return result


class Sub_CoT:
    """
    The smallest unit that decomposes the overall task,
    including a starting node representing the known conditions and
    an ending node representing the phased goal
    """

    # 为每个Thought实例分配唯一的证书ID
    _ids: Iterator[int] = itertools.count(0)

    # state是thought的初始状态, 默认为None, 是一个字典类型
    def __init__(self, known_condition, goal, additional_known_conditions=None) -> None:
        """
        Initializes a new Thought instance with a state and various default flags.
        """
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)  # 创建一个名为Thought的日志对象, 记录thought日志信息
        self.id: int = next(Sub_CoT._ids)  # 为每个thought实例分配一个唯一的整数ID
        self._score: float = 0.0  # thought的评分
        self.compared_to_ground_truth: bool = False  # thought的结果是否与真实值进行了对比

        self.decomposability: bool = True  # Sub_CoT的可分解性
        self.decomposed: bool = False  # Sub_CoT是否已经被分解
        # Sub_CoT的已知条件 key: <initial> 初始的已知条件, <1...n> 对应前面每一步的操作说名
        self.known_condition: list[str] = known_condition
        self.goal: str = goal  # Sub_CoT的已知条件
        if additional_known_conditions is None:
            self.additional_known_conditions = []
        else:
            self.additional_known_conditions: List[str] = additional_known_conditions
        self.middles: List[MiddleGoal] = []  # 生成的中间目标
        self.middle: Optional[MiddleGoal] = None  # 最终的中间目标
        self.order: List = []  # 记录该sub_cot从何而来
        if self.id == 0:
            self.order = [0]
        self.result: Dict = {}

    def Decomposability(self, lm: str, prompter: Prompter, parser: Parser,
                        decomposability_function=None):
        """
        判断当前sub_cot是否可以继续被分解
        """
        if decomposability_function is not None:
            self.decomposability = decomposability_function(self.known_condition, self.goal)
        else:
            prompt = prompter.decomposition_necessity_prompt(known_information=self.known_condition,
                                                             additional_known_information=self.additional_known_conditions,
                                                             goal=self.goal)
            agent = Agent(model=lm)
            responses, conversationHistory = agent.conversation(prompt)
            print("SubCoT.Decomposability: ", responses)
            result = parser.parse_decomposition_necessity_answer(responses)
            self.decomposability = not result["result"]

    # 根据num_response设置不同的温度进行温度采样
    def decomposition(self, lm: str, prompter: Prompter, parser: Parser,
                      num_response: int = 1):
        """
        分解sub_cot.
        """
        # 判断当前sub_cot能否被分解
        if not self.decomposability:
            return
        # 如果可分解则进行分解
        while not self.middles:
            prompt = prompter.generate_key_goal_prompt(self.known_condition, self.additional_known_conditions,
                                                       self.goal)
            agent = Agent(model=lm)
            responses, conversationHistory = agent.conversation(prompt, num_response=num_response)
            print("SubCoT.decomposition: ", responses)
            prompt = prompter.Create_known_information(responses)
            responses, conversationHistory = agent.conversation(prompt)
            print("Create_known_conditions: ", responses)
            result = parser.parse_create_known_information_answer(responses)
            for m in result:
                previous_goal = m["question"]
                next_conditions = m["known information"]
                self.middles.append(MiddleGoal(previous_goal, next_conditions))
            self.logic_detection(lm, prompter, parser)
        self.Score(lm, prompter, parser)
        self.decomposed = True

    def logic_detection(self, lm: str, prompter: Prompter, parser: Parser) -> None:
        Questions = []
        for m in self.middles:
            Questions.append(m.previous_goal)

        # 无关性检测
        results = [{}] * len(Questions)  # 创建与输入大小相同的结果占位列表
        known_condition = self.known_condition.copy()
        additional_known_conditions = self.additional_known_conditions.copy()
        with ThreadPoolExecutor() as logic_detection_executor:
            # 使用enumerate跟踪消息的索引
            futures = {
                logic_detection_executor.submit(Logic_dependency_detection, lm, prompter, parser, known_condition,
                                                additional_known_conditions, question): idx
                for idx, question in enumerate(Questions)
            }
            for future in as_completed(futures):
                idx = futures[future]  # 获取任务对应的输入索引
                result = future.result()  # 解包结果
                results[idx] = result  # 按索引存储结果

        print("无关性检测:", results)

        del_middles = set()
        n = 0
        for r in results:
            if not r["result"]:
                del_middles.add(n)
            n += 1
        self.middles = [x for i, x in enumerate(self.middles) if i not in del_middles]

        if not self.middles:
            return

        # 直接重复求解检测
        Questions = []
        for m in self.middles:
            Questions.append(m.previous_goal)

        results = [{}] * len(Questions)  # 创建与输入大小相同的结果占位列表
        goal = self.goal
        with ThreadPoolExecutor() as logic_detection_executor:
            # 使用enumerate跟踪消息的索引
            futures = {
                logic_detection_executor.submit(Direct_logical_duplicate_detection, lm, prompter, parser, known_condition, additional_known_conditions, goal, question): idx
                for idx, question in enumerate(Questions)
            }
            for future in as_completed(futures):
                idx = futures[future]  # 获取任务对应的输入索引
                result = future.result()  # 解包结果
                results[idx] = result  # 按索引存储结果

        print("直接重复: ", results)

        del_middles = set()
        n = 0
        for r in results:
            if not r["result"]:
                del_middles.add(n)
            n += 1
        self.middles = [x for i, x in enumerate(self.middles) if i not in del_middles]

        if not self.middles:
            return

        # 间接重复求解检测
        Information = []
        for m in self.middles:
            Information.append(m.next_conditions)

        results = [{}] * len(Information)  # 创建与输入大小相同的结果占位列表
        with ThreadPoolExecutor() as logic_detection_executor:
            # 使用enumerate跟踪消息的索引
            futures = {
                logic_detection_executor.submit(Indirect_logical_duplication_detection, lm, prompter, parser, known_condition, additional_known_conditions, goal, information): idx
                for idx, information in enumerate(Information)
            }
            for future in as_completed(futures):
                idx = futures[future]  # 获取任务对应的输入索引
                result = future.result()  # 解包结果
                results[idx] = result  # 按索引存储结果

        print("间接重复: ", results)

        del_middles = set()
        n = 0
        for r in results:
            if not r["result"]:
                del_middles.add(n)
            n += 1
        self.middles = [x for i, x in enumerate(self.middles) if i not in del_middles]

    def find_best_sub_task(self, s: str) -> int:
        # 去除输入字符串中的标点符号
        s = s.translate(str.maketrans('', '', string.punctuation))

        # 遍历 MiddleGoal 列表查找匹配的 previous_goal
        for index, middle_goal in enumerate(self.middles):
            # 去除 previous_goal 的标点符号
            next_conditions = middle_goal.next_conditions.translate(str.maketrans('', '', string.punctuation))
            # 比较去掉标点符号后的字符串
            if s == next_conditions:
                return index
        # 如果没有匹配项，返回 -1
        return -1

    def Score(self, lm: str, prompter: Prompter, parser: Parser,
              num_response: int = 1):
        """
        为生成的每个结果打分并保留最好的结果
        """
        # 如果没有生成中间目标则不予打分
        if len(self.middles) == 0:
            print("还没有为当前sub_Cot生成中间目标")
            return

        if len(self.middles) == 1:
            self.middle = self.middles[0]
            self.middles = {}
            return

        # 如果已经生成了中间目标, 则就是要正经调用LLM给sub_cot打分了
        Middles = []
        for m in self.middles:
            Middles.append(m.next_conditions)
        prompt = prompter.score_prompt(self.known_condition, self.additional_known_conditions, Middles,
                                       self.goal)
        agent = Agent(model=lm)
        responses, conversationHistory = agent.conversation(prompt)
        print("SubCoT.Score: ", responses)

        score = parser.parse_score_answer(responses)
        best_sub_task = max(score, key=lambda x: x["result"])["information"]
        # 搜索best_sub_task字符串与self.middles中哪一个MiddleGoal.previous_goal相匹配
        best_middle_index = self.find_best_sub_task(best_sub_task)
        self.middle = self.middles[best_middle_index]
        self.middles = {}

    def inference_sub_cot(self, lm: str, prompter: Prompter,
                          parser: Parser,
                          num_response: int = 1, task_type="text") -> None:
        prompt = prompter.inference_prompt(self.known_condition,
                                           self.additional_known_conditions, self.goal)
        agent = Agent(model=lm)
        responses, conversationHistory = agent.conversation(prompt)
        print("Sub_CoT.inference_sub_cot: ", responses)

        self.result = parser.parse_inference_answer(responses)

    def Complete_the_known(self, lm: str, prompter: Prompter,
                           parser: Parser, pre_result: List[str],
                           num_response: int = 1) -> None:
        prompt = prompter.Answer_to_known_prompt(pre_result, self.additional_known_conditions)
        agent = Agent(model=lm)
        responses, conversationHistory = agent.conversation(prompt)
        print("Sub_CoT.Complete_the_known: ", responses)

        self.additional_known_conditions = parser.parse_Answer_to_known_answer(responses)
