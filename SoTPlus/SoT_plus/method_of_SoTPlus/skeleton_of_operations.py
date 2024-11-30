from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, wait
from itertools import dropwhile
from typing import Dict, Callable, List

# from SoT_plus.fundation_models.Llama import Agent
from SoT_plus.fundation_models.GPT import Agent
from SoT_plus.method_of_SoTPlus.sub_cot import Sub_CoT
from SoT_plus.parser import Parser
from SoT_plus.prompter import Prompter


class SkelentonOfThought:
    """
    This class is used to store the complete reasoning chain during the reasoning process.
    """

    def __init__(self) -> None:
        """
        Initializes a new Graph of Operations instances with empty operations, roots, and leaves.
        The roots are the entry points in the graph with no predecessors.
        The leaves are the exit points in the graph with no successors.
        """
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.cots: Dict = {}  # 存储要执行的sub_cot对象 key: sub_cot.id value: sub_cot
        self.order: Dict = {}  # 存储sub_cot的执行顺序 key: sub_cot.id value: order
        self.type: Dict = {}  # 存储每个sub_cot的任务类别 key: sub_cot.id value:("text", "math&logic")
        self.next_dec: int = 0  # 相当于一个指针, 指向self.cots中下一个需要分解的sub_cot的key

        self.decomposition_stage: bool = False  # 表示完整的cot是否已经生成
        self.Priority_Adjustment_stage: bool = False  # 表示每个sub_cot执行优先级的顺序已调整完毕
        self.Classification_stage: bool = False  # 表示每个sub_cot的任务类型已分类完毕
        self.final_result: Dict = {}
        self.pre_result = {}

    def next__sub_cot(self, parallel: bool = False) -> List[Sub_CoT] | None:
        """
        移动self.next_dec指针指向下一个self.cots中需要被分解的sub_cot对象的key,
        """
        if parallel:
            del_cache = []  # 记录需要删除的cot
            dec_cache = []  # 记录需要分解的cot
            # 检查当前self.next_dec指向的sub_key能否被分解
            for key, current_sub_cot in self.cots.items():
                # 如果sub_cot是熟的, 就分解并添加
                if current_sub_cot.decomposed:
                    del_cache.append(current_sub_cot)
                # 记录最后一个死的, 最后就用self.next_dec指向最后一个死的
                elif current_sub_cot.decomposability is not True:
                    continue
                else:
                    dec_cache.append(current_sub_cot)
            if len(del_cache) == 0:
                if len(dec_cache) == 0:
                    self.decomposition_stage = True
                    return
                else:
                    return dec_cache
            # 添加新cot到self.cots, 并从self.cots中删除del_cache中的cot
            for cot in del_cache:
                new_sub_cot_0 = Sub_CoT(cot.known_condition, cot.middle.previous_goal,
                                        cot.additional_known_conditions.copy())
                new_condition = cot.additional_known_conditions.copy()
                new_condition.append(cot.middle.next_conditions)
                new_sub_cot_1 = Sub_CoT(cot.known_condition, cot.goal, new_condition)

                new_sub_cot_0.order = cot.order.copy()
                new_sub_cot_1.order = cot.order.copy()

                new_sub_cot_0.order.append(0)
                new_sub_cot_1.order.append(1)

                self.cots[new_sub_cot_0.id] = new_sub_cot_0
                self.cots[new_sub_cot_1.id] = new_sub_cot_1
                del self.cots[cot.id]

                dec_cache.append(new_sub_cot_0)
                dec_cache.append(new_sub_cot_1)
            return dec_cache
        else:
            Find_next_dec = False
            while Find_next_dec is not True:
                del_cache = []  # 记录需要删除的cot
                # 检查当前self.next_dec指向的sub_key能否被分解
                for key in dropwhile(lambda k: k != self.next_dec, self.cots):
                    current_sub_cot = self.cots[key]
                    # 如果sub_cot是熟的, 就分解并添加
                    if current_sub_cot.decomposed:
                        del_cache.append(current_sub_cot)
                    # 如果是生的, 就用self.next_dec指向它
                    elif current_sub_cot.decomposability:
                        self.next_dec = key
                        Find_next_dec = True
                if Find_next_dec is not True:
                    self.next_dec = list(self.cots.keys())[-1]
                if len(del_cache) == 0:
                    break
                # 添加新cot到self.cots, 并从self.cots中删除del_cache中的cot
                for cot in del_cache:
                    new_sub_cot_0 = Sub_CoT(cot.known_condition, cot.middle.previous_goal, cot.additional_known_conditions.copy())
                    new_condition = cot.additional_known_conditions.copy()
                    new_condition.append(cot.middle.next_conditions)
                    new_sub_cot_1 = Sub_CoT(cot.known_condition, cot.goal, new_condition)

                    new_sub_cot_0.order = cot.order.copy()
                    new_sub_cot_1.order = cot.order.copy()

                    new_sub_cot_0.order.append(0)
                    new_sub_cot_1.order.append(1)

                    self.cots[new_sub_cot_0.id] = new_sub_cot_0
                    self.cots[new_sub_cot_1.id] = new_sub_cot_1
                    del self.cots[cot.id]
            # 如果Find_next_dec, 证明已经找到下一个需要分解的cot, 可以返回
            if Find_next_dec:
                return

            # 如果前面没有return, 则证明self.cots中所有的sub_cot都不可分解, 即针对问题的整个cot已经生成完毕
            self.decomposition_stage = True

    def Classification(self, sub_cot: Sub_CoT, lm: str, prompter: Prompter,
                       parser: Parser,
                       classification_function=None):
        """
        Classify the task type for each sub_cot, store the classification result in self.type
        """
        if classification_function is not None:
            self.logger.debug(
                "Using classification function %s to determine the task type of the current sub_cot",
                classification_function,
            )
            self.type[sub_cot.id] = classification_function(sub_cot.known_condition,
                                                            sub_cot.additional_known_conditions, sub_cot.goal)
        else:
            prompt = prompter.subtask_classification_prompt(sub_cot.known_condition,
                                                            sub_cot.additional_known_conditions, sub_cot.goal)
            self.logger.debug("Prompt for LM: %s", prompt)
            agent = Agent(model=lm)
            responses, conversationHistory = agent.conversation(prompt)
            print("SkelentonOfThought.Classification: ", responses)
            self.logger.debug("Responses from LM: %s", responses)
            self.type[sub_cot.id] = parser.parse_subtask_classification_answer(responses)

    def Classification_for_all(self, lm: str, prompter: Prompter,
                               parser: Parser,
                               classification_function: Callable = None):
        for key, value in self.cots.items():
            current_sub_cot = value
            self.Classification(current_sub_cot, lm, prompter,
                                parser,
                                classification_function)
        self.Classification_stage: bool = True

    def Priority_Adjustment(self, sub_cot: Sub_CoT, lm: str, prompter: Prompter,
                            parser: Parser) -> None:
        """
        通过依次比较每个sub_cot对象中goal和additional_known_conditions中每个子目标的依赖关系来判定该sub_cot执行优先级
        """
        n = 0
        for sub_target in reversed(sub_cot.additional_known_conditions):
            prompt = prompter.dependency_detection_prompt(sub_cot.known_condition, sub_target, sub_cot.goal)
            agent = Agent(model=lm)
            responses, conversationHistory = agent.conversation(prompt)
            print("SkelentonOfThought.Priority_Adjustment: ", responses)
            result = parser.parse_dependency_detection_answer(responses)
            if result["result"] is not True:
                n += 1
            else:
                break
        # 计算最终的优先级
        sub_cot.additional_known_conditions = sub_cot.additional_known_conditions[:-n]
        self.order[sub_cot.id] = len(sub_cot.additional_known_conditions)

    def Priority_Adjustment_for_all(self, lm: str, prompter: Prompter,
                                    parser: Parser,
                                    num_response: int = 1) -> None:
        """
        对self.cots中的所有sub_cot的执行优先级进行调整
        """
        for key in reversed(self.cots):
            current_sub_cot = self.cots[key]
            self.Priority_Adjustment(current_sub_cot, lm, prompter, parser)
        self.Priority_Adjustment_stage: bool = True

    # 最后推断每个推理链时, 也可以选择性使用self-consistency方法, 这里可以多一个参数Multiple_cot = "self-consistency"或"生成多个结果让LLM选择一个"
    def inference(self, lm: str, prompter: Prompter,
                  parser: Parser,
                  parallel: bool = False) -> None:
        """
        对所有sub_cot的结果进行推理并保存结果
        """
        # 初始化所有sub_cot的执行优先层级
        Order: Dict = {}
        for key, value in self.order.items():  # key为id, value为优先级
            if value in Order:
                Order[value].append(key)
            else:
                Order[value] = [key]
        Order = dict(sorted(Order.items()))

        # 开始推理每一个sub_cot的结果
        n = 1
        for key, value in Order.items():
            if parallel:
                sub_cots = []
                for s in value:
                    sub_cot = self.cots[s]
                    sub_cots.append(sub_cot)

                with ThreadPoolExecutor() as inference_executor:
                    futures = [
                        inference_executor.submit(sub_cot.inference_sub_cot, lm, Prompter, parser)
                        for sub_cot in sub_cots
                    ]

                    # 等待所有线程完成
                    wait(futures)

                # 将答案共享给下一优先级的sub_cot
                if n < len(Order):
                    # 提取上一个优先级的所有sub_cot的结果组成一个list[str]
                    pre_result = []
                    for sub_cot in value:
                        pre_result.append(sub_cot.result["result"])
                    # 将下一层的每个sub_cot的addition_known_condition与list[str]匹配
                    keys = list(Order.keys())  # 将键转换为列表
                    next_sub_cots = Order[keys[n]]  # 得到下一层的sub_cots
                    with ThreadPoolExecutor() as inference_executor:
                        futures = [
                            inference_executor.submit(sub_cot.Complete_the_known, lm, Prompter, parser)
                            for sub_cot in next_sub_cots
                        ]

                        # 等待所有线程完成
                        wait(futures)
                n += 1
            else:
                for s in value:
                    sub_cot = self.cots[s]
                    sub_cot.inference_sub_cot(lm, prompter, parser)

    def final_output(self, lm: str, prompter: Prompter, parser: Parser) -> None:
        """
        总推理, 基于每个分结果生成最终答案
        """
        Cot: Dict = {i: None for i in range(1, next(reversed(self.cots)) + 1)}
        # 排序: 根据sub_cot.order码来加入Cot
        for key, value in self.cots.items():
            order = 0
            for i in range(0, len(value.order) - 1):
                order += 2 ^ i
                if value.order[i] == 1:
                    order += 2
            if value.order[-1] == 0:
                order += 1
            if value.order[-1] == 1:
                order += 2
            Cot[order] = value
        Cot = {k: v for k, v in Cot.items() if v is not None}

        question = Cot[next(iter(Cot))].known_condition
        question = " ".join(question) + Cot[next(reversed(Cot))].goal

        final_prompt = {"question": question, "Reasoning steps": []}

        for key, value in Cot.items():
            final_prompt["Reasoning steps"].append(value.result["explain"])

        # 生成最终的答案
        prompt = prompter.summarization_prompt(final_prompt)
        agent = Agent(model=lm)
        responses, conversationHistory = agent.conversation(prompt)
        print("SkelentonOfThought.final_output: ", responses)
        self.final_result = parser.parse_summarization_answer(responses)
