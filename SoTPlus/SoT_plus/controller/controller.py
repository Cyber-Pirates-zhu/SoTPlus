import logging
# from SoT_plus.fundation_models.Llama import Agent
from SoT_plus.fundation_models.GPT import Agent
from SoT_plus.method_of_SoTPlus import SkelentonOfThought, Sub_CoT
from SoT_plus.parser import Parser
from SoT_plus.prompter import Prompter
from concurrent.futures import ThreadPoolExecutor, wait


class Controller:
    """
    Controller class to manage the execution flow of the Graph of Operations,
    generating the Graph Reasoning State.
    This involves language models, graph operations, prompting, and parsing.
    """

    def __init__(
            self,
            lm: str,  # lm参数: 用的LLM, 例如GPT, llama...
            SoT: SkelentonOfThought,  # ChainOfThought: 完整的推理链
            prompter: Prompter,  # ChainOfThought信息 --> prompt(从而输入到LLM)
            parser: Parser,  # prompt --> ChainOfThought信息(从而储存并推理)
            Parallel: bool = False
    ) -> None:  # 表明参数的返回类型, __init__返回类型一般为None
        """
        控制器: 相当于一个协调者, 负责各个对象的调用与之间的信息交流
        """
        self.logger = logging.getLogger(self.__class__.__module__)  # 日志记录器，用于记录执行状态和信息。
        self.lm = lm
        self.SoT = SoT
        self.prompter = prompter
        self.parser = parser
        self.run_executed = False  # 布尔变量，标记 run 方法是否被执行过。
        self.Parallel = Parallel

    def execute_parallel_dec_methods(self, sub_cot: Sub_CoT):
        return (
            sub_cot.Decomposability(self.lm, self.prompter, self.parser),
            sub_cot.decomposition(self.lm, self.prompter, self.parser, num_response=3),
        )

    def execute_priority_classification_methods(self, sub_cot: Sub_CoT):
        return (
            self.SoT.Priority_Adjustment(sub_cot, self.lm, self.prompter, self.parser)
            # self.SoT.Classification(sub_cot, self.lm, self.prompter, self.parser),
        )

    def run(self, question: str, num_response: int = 1) -> None:
        """
        运行整个SoT+算法
        """
        # 生成初始的CoT, 加入到self.SoT.cots当中
        prompt = self.prompter.Initialize_CoT_prompt(question)
        agent = Agent(model=self.lm)
        responses, conversationHistory = agent.conversation(prompt)
        r = self.parser.parse_Initialize_CoT_answer(responses[0])
        first_sub_cot = Sub_CoT(r["known information"], r["target"])
        self.SoT.cots[first_sub_cot.id] = first_sub_cot

        print("parse_Initialize_CoT_answer: ", r)
        print("___________________________________________")
        # 二次拆分
        # known_conditions = []
        # for sub_target in r["known information"]:
        #     prompt = self.prompter.Initialize_CoT_prompt(sub_target + r["target"])
        #     responses, conversationHistory = self.lm.conversation(prompt)
        #     rr = self.parser.parse_Initialize_CoT_answer(responses[0])
        #     for t in rr["known information"]:
        #         known_conditions.append(t)
        # known_conditions = list(dict.fromkeys(known_conditions))
        # first_sub_cot = Sub_CoT(known_conditions, r["target"])
        # self.SoT.cots[first_sub_cot.id] = first_sub_cot
        # print("known_conditions: ", known_conditions)

        # 开始分解
        n = 0
        while self.SoT.decomposition_stage is not True:
            if self.Parallel:
                current_dec_cots = self.SoT.next__sub_cot(parallel=self.Parallel)
                if self.SoT.decomposition_stage:
                    break
                if len(current_dec_cots) > 6:  # 限制线程数量, 避免超负载
                    break
                '''验证分解性, 分解, 逻辑检测, 打分所有sub_cot'''
                print("%%%%%%: ", len(current_dec_cots))
                with ThreadPoolExecutor() as controller_executor:
                    futures = [
                        controller_executor.submit(self.execute_parallel_dec_methods, sub_cot)
                        for sub_cot in current_dec_cots
                    ]

                    # 等待所有线程完成
                    wait(futures)
                # with ThreadPoolExecutor() as controller_executor:
                #     for sub_cot in current_dec_cots:
                #         future = controller_executor.submit(self.execute_parallel_dec_methods, sub_cot)
                #         future.result()  # 阻塞主线程，等待当前任务完成

                for key, cot in self.SoT.cots.items():
                    print("cot.id: ", cot.id)
                    print("cot.decomposability: ", cot.decomposability)
                    print("cot.known_condition: ", cot.known_condition)
                    print("cot.additional_known_conditions: ", cot.additional_known_conditions)
                    if cot.middle is not None:
                        print("cot.middle.previous_goal: ", cot.middle.previous_goal)
                        print("cot.middle.next_conditions: ", cot.middle.next_conditions)
                    print("cot.goal: ", cot.goal)
                    print(f"{n}--------------------------")
                n += 1
            else:
                self.SoT.next__sub_cot()  # 找到self.SoT.cots中可分解的对象
                if self.SoT.decomposition_stage:
                    break
                current_dec_cot = self.SoT.cots[self.SoT.next_dec]
                current_dec_cot.Decomposability(self.lm, self.prompter, self.parser)
                if current_dec_cot.decomposability:
                    current_dec_cot.decomposition(self.lm, self.prompter, self.parser, num_response=3)
                    current_dec_cot.Score(self.lm, self.prompter, self.parser, num_response=1)
                    self.SoT.cots[current_dec_cot.id] = current_dec_cot

        # if self.Parallel:
        #     # 调整优先级和分类
        #     with ThreadPoolExecutor() as controller_executor:
        #         futures = [
        #             controller_executor.submit(self.execute_priority_classification_methods, sub_cot)
        #             for sub_cot in self.SoT.cots
        #         ]
        #
        #         # 等待所有线程完成
        #         wait(futures)
        #
        #     # 按优先级执行
        #     self.SoT.inference(self.lm, self.prompter, self.parser, parallel=True)
        #
        # else:
        #     # 调整顺序
        #     self.SoT.Priority_Adjustment_for_all(self.lm, self.prompter, self.parser)
        #
        #     # 执行
        #     self.SoT.inference(self.lm, self.prompter, self.parser)
        #
        # # 汇总得出最终答案
        # self.SoT.final_output(self.lm, self.prompter, self.parser)

        # 保存结果到json格式
        # self.save_answer()

    def save_answer(self) -> None:
        """
        保存数据, 用于之后的画图
        """
        pass
