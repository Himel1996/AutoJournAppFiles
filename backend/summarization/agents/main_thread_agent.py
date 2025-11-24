from typing import Dict, List

from summarization.agents.agent import Agent
from summarization.models.summarizer import SummarizationModel


class MainThreadAgent(Agent):
    def __init__(self, summarizer_model: SummarizationModel):
        super(MainThreadAgent, self).__init__(summarizer_model)

    @staticmethod
    def get_main_thread(conv_root: Dict) -> List[Dict]:
        main_thread = [conv_root]

        # no replies to head tweet. The conversation is only one tweet
        if "replies" not in conv_root or ("replies" in conv_root and len(conv_root["replies"]) == 0):
            return main_thread

        # Iterate over replies of head tweet and add it to the main thread
        for reply in conv_root["replies"]:
            main_thread.append(reply)

        return main_thread

    def run_conv(self, conv_root: Dict) -> str:
        main_thread = self.get_main_thread(conv_root)
        tweets_list = [tweet["username"] + ":" + tweet["text"] for tweet in main_thread]
        summary = self.summarizer_model.summarize(tweets_list)
        return summary
