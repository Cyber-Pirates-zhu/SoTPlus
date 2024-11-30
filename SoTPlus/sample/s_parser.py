import ast
import json
import re
from typing import Dict, List
from SoT_plus.parser import Parser


def remove_extra_content(texts: str) -> str:
    texts = texts.replace("\n", "")
    match = re.search(r'\{.*}', texts)
    return match.group()

class s_Parser(Parser):
    def parse_Initialize_CoT_answer(self, texts: str) -> Dict:
        texts = remove_extra_content(texts)
        return json.loads(texts)

    def generate_key_goal_answer(self, texts: str) -> str:
        return texts

    def parse_create_known_information_answer(self, texts_list: List[str]) -> list[Dict]:
        responses = texts_list[0]
        json_match = re.search(r'\[.*?]', responses, re.DOTALL)
        json_str = json_match.group()  # 提取到的 JSON 字符串
        return json.loads(json_str)

    def Logic_dependency_detection_answer(self, texts: list[str]) -> Dict:
        """
        Parse the response from the language model for a score prompt.
        """
        texts = remove_extra_content(texts[0])
        result = json.loads(texts)
        return result

    def Direct_logical_duplicate_detection_answer(self, texts: list[str]) -> Dict:
        """
        Parse the response from the language model for a score prompt.
        """
        texts = remove_extra_content(texts[0])
        return json.loads(texts)

    def Indirect_logical_duplication_detection_answer(self, texts: list[str]) -> Dict:
        """
        Parse the response from the language model for a score prompt.
        """
        texts = remove_extra_content(texts[0])
        return json.loads(texts)

    def parse_score_answer(self, texts_list: list[str]) -> list[Dict]:
        responses = texts_list[0]
        json_match = re.search(r'\[.*?]', responses, re.DOTALL)
        json_str = json_match.group()  # 提取到的 JSON 字符串
        return json.loads(json_str)

    def parse_decomposition_necessity_answer(self, texts_list: list[str]) -> Dict:
        texts = remove_extra_content(texts_list[0])
        return ast.literal_eval(texts)

    def parse_subtask_classification_answer(self, texts: str) -> str:
        return texts

    def parse_dependency_detection_answer(self, texts: str) -> Dict:
        texts = remove_extra_content(texts[0])
        return json.loads(texts)

    def parse_inference_answer(self, texts: str) -> Dict:
        texts = remove_extra_content(texts[0])
        return json.loads(texts)

    def parse_summarization_answer(self, texts: str) -> Dict:
        texts = remove_extra_content(texts[0])
        return json.loads(texts)

    def parse_Answer_to_known_answer(self, texts: str) -> Dict:
        responses = texts[0]
        json_match = re.search(r'\[.*?]', responses, re.DOTALL)
        json_str = json_match.group()  # 提取到的 JSON 字符串
        return json.loads(json_str)
