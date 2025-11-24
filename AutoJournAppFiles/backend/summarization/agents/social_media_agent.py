from typing import Dict, List

from summarization.agents.agent import Agent
from summarization.models.summarizer import SummarizationModel


class SocialMediaAgent(Agent):
    def __init__(self, summarizer_model: SummarizationModel):
        super(SocialMediaAgent, self).__init__(summarizer_model)

    def run_conv(self, conv_root: list) -> str:
        return self.summarizer_model.summarize(conv_root)
