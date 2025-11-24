from typing import Dict

from summarization.agents.agent import Agent
from summarization.models.summarizer import SummarizationModel


class ConvTreeAgent(Agent):
    def __init__(self, summarizer_model: SummarizationModel):
        super(ConvTreeAgent, self).__init__(summarizer_model)

    def __dfs(self, root: dict) -> (str, str):
        text = root["text"]

        # no replies so return processed tweet
        if "replies" not in root or ("replies" in root and len(root["replies"]) == 0):
            return root["username"], text

        thread = [root["username"] + ":" + text]  # add head tweet of the thread

        # Iterate over replies and summarize child threads
        for reply in root["replies"]:
            username, child_thread_summary = self.__dfs(reply)
            thread.append(username + ":" + child_thread_summary)

        # summarize the thread
        thread_summary = self.summarizer_model.summarize(thread)
        return root["username"], thread_summary

    def run_conv(self, conv_root: Dict) -> str:
        _, summary = self.__dfs(conv_root)
        return summary
