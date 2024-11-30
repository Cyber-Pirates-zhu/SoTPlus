from SoT_plus.controller import Controller
from SoT_plus.method_of_SoTPlus import SkelentonOfThought
from sample import s_Parser
from sample import s_Prompter
from datasets import load_dataset
import openai

openai.api_key = "api_key"
openai.api_base = "https://api.openai.com/v1"

if __name__ == "__main__":
    SoT = SkelentonOfThought()
    prompter = s_Prompter()
    parser = s_Parser()
    controller = Controller(lm="gpt-4", SoT=SoT, prompter=prompter, parser=parser, Parallel=True)

    # 加载 main 数据集
    # main_ds = load_dataset("openai/gsm8k", "main")
    # for i in range(0, 10):
    #     question = main_ds['train'][i]['question']
    #     print(question)  # 查看第一条数据
    #     controller.run(question=question)

    question = (
        "Mrs.Lim milks her cows twice a day. Yesterday morning, she got 68 gallons of milk and in the evening, she got "
        "82 gallons. This morning, she got 18 gallons fewer than she had yesterday morning. After selling some gallons "
        "of milk in the afternoon, Mrs.Lim has only 24 gallons left. How much was her revenue for the milk if each "
        "gallons costs $3.50?"
    )

    controller.run(question=question)
    print("final answer: ", SoT.final_result)
