from abc import ABC, abstractmethod
from typing import List, Dict
from summarization.models.summarizer import SummarizationModel


class Agent(ABC):

    def __init__(self, summarizer_model: SummarizationModel):
        self.summarizer_model = summarizer_model

    @abstractmethod
    def run_conv(self, conv_root: list) -> str:
        pass

    def run_all(self, conv_list: List[Dict[str, List[str]]]) -> Dict[str, str]:
        summaries = []  # conversation summaries
        for conv_dict in conv_list:
            conv_root = conv_dict['dialogue']
            conv_id = conv_dict['id']
            conv_summary = self.run_conv([conv_root])
            summaries.append({'id': conv_id, 'summary': conv_summary})

        return summaries

    def run_all_topic_aware(self, dict_topic_sentences: Dict[str, List[str]]) -> Dict[str, str]:
        topic_summaries = {}
        for topic, sentences in dict_topic_sentences.items():
            conv_summary = self.run_conv(sentences)
            topic_summaries[topic] = conv_summary

        return topic_summaries
