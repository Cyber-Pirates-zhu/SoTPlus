from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List


class Prompter(ABC):
    """
    Abstract base class that defines the interface for all prompters.
    Prompters are used to generate the prompts for the language models.
    """

    @abstractmethod
    def Initialize_CoT_prompt(self, question: str) -> str:
        """
        Decompose user input questions into known conditions and goals
        """
        pass

    @abstractmethod
    def generate_key_goal_prompt(self, known_information: List[str], additional_known_information: List[str],
                                 goal: str) -> str:
        """
        Generate a most critical intermediate goal Based on the initial
        conditions and termination goals of each sub-CoT.
        """
        pass

    @abstractmethod
    def Create_known_information(self, sub_questions: List[str]) -> str:
        pass

    @abstractmethod
    def Logic_dependency_detection_prompt(self, known_information: List[str], additional_known_information: List[str],
                                          question: str) -> str:
        pass

    @abstractmethod
    def Direct_logical_duplicate_detection_prompt(self, known_information: List[str], additional_known_information: List[str],
                                                  middle: str) -> str:
        pass

    @abstractmethod
    def Indirect_logical_duplication_detection_prompt(self, known_information: List[str],
                                                      additional_known_information: List[str],
                                                      middle: str,
                                                      goal: str) -> str:
        pass

    @abstractmethod
    def score_prompt(self, known_information: List[str], additional_known_information: List[str], middle: List[str],
                     goal: str) -> str:
        """
        This operation is used after the generate_key_goal_prompt call to make
        LLM score each generated result to select the best result.
        """
        pass

    @abstractmethod
    def decomposition_necessity_prompt(self, known_information: list[str], additional_known_information: List[str],
                                       goal: str) -> str:
        """
        Enable LLM to determine whether the input sub-CoT needs to be further decomposed

        :return: The score prompt.
        """
        pass

    @abstractmethod
    def subtask_classification_prompt(self, known_information: List[str], additional_known_information: List[str],
                                      goals: str) -> str:
        """
        Use LLM to determine which category the reasoning task of each sub-CoT belongs to.
        """
        pass

    @abstractmethod
    def dependency_detection_prompt(self, known_information: List[str], sub_target: str, goal: str) -> str:
        """
        Check the dependency between two sub-CoTs
        """
        pass

    @abstractmethod
    def inference_prompt(self, known_information: List[str], additional_known_information: List[str],
                         goals: str) -> str:
        """
        Check the dependency between two sub-CoTs
        """
        pass

    @abstractmethod
    def summarization_prompt(self, final_prompt: Dict) -> str:
        """
        Generate the final answer based on the results of all sub-CoTs.
        """
        pass

    @abstractmethod
    def Answer_to_known_prompt(self, pre_result: List[str], additional_known_information: List[str]) -> str:
        pass
