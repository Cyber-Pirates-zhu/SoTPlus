from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List


class Parser(ABC):
    """
    Abstract base class that defines the interface for all parsers.
    Parsers are used to parse the responses from the language models.
    """

    @abstractmethod
    def parse_Initialize_CoT_answer(self, texts: str) -> Dict:
        """
        Parse the response from the language model for an Initialize_CoT prompt.
        """
        pass

    @abstractmethod
    def generate_key_goal_answer(self, texts: str) -> str:
        pass

    @abstractmethod
    def parse_create_known_information_answer(self, texts_list: List[str]) -> Dict:
        pass

    @abstractmethod
    def Logic_dependency_detection_answer(self, texts_list: List[str]) -> Dict:
        pass

    @abstractmethod
    def Direct_logical_duplicate_detection_answer(self, texts_list: List[str]) -> Dict:
        pass

    @abstractmethod
    def Indirect_logical_duplication_detection_answer(self, texts_list: List[str]) -> Dict:
        pass

    @abstractmethod
    def parse_score_answer(self, texts: str) -> Dict:
        """
        Parse the response from the language model for a score prompt.
        """
        pass

    @abstractmethod
    def parse_decomposition_necessity_answer(self, texts: str) -> Dict:
        """
        Parse the response from the language model for a decomposition_necessity prompt.

        :param state: The thought state used to generate the prompt.
        :param texts: The responses to the prompt from the language model.
        :return: Whether the thought state is valid or not.
        """
        pass

    @abstractmethod
    def parse_subtask_classification_answer(self, texts: str) -> Dict:
        """
        Parse the response from the language model for a subtask_classification prompt.

        :param states: The thought states used to generate the prompt.
        :param texts: The responses to the prompt from the language model.
        :return: The scores for the thought states.
        """
        pass

    @abstractmethod
    def parse_dependency_detection_answer(self, texts: str) -> Dict:
        """
        Parse the response from the language model for a dependency_detection prompt.
        """
        pass

    @abstractmethod
    def parse_inference_answer(self, texts: str) -> Dict:
        """
        Check the dependency between two sub-CoTs
        """
        pass

    @abstractmethod
    def parse_summarization_answer(self, texts: str) -> Dict:
        """
        Parse the response from the language model for a summarization prompt.
        """
        pass

    @abstractmethod
    def parse_Answer_to_known_answer(self, texts: str) -> Dict:
        pass
