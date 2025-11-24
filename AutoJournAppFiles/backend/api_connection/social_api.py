import json
from summarization.models.samsum import Samsum, MessageThread


class SocialAPI:
    def __init__(self):
        pass

    def parse_all_messages(self, message_threads: list[MessageThread]) -> list[Samsum]:
        samsums = []
        for message_thread in message_threads:
            samsums.append(self.parse_message(
                message_thread.messages, message_thread.id))
        return samsums

    def parse_all_messages_json(self, message_threads: list[MessageThread]) -> list[dict]:
        return [self.parse_message_json(messages=message_thread.messages, id=message_thread.id) for message_thread in message_threads]

    # for testing purposes
    def export_query_as_json(self, query: str, channel_limit=5, message_limit=10, filename="results.json") -> None:
        result = self.query(query, channel_limit, message_limit)
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(self.parse_all_messages_json(
                result), file, ensure_ascii=False)
