class Samsum:
    def __init__(self, id, summary, dialogue):
        self.id = id
        self.summary = summary
        self.dialogue = dialogue

    def to_json(self):
        return {"id": self.id, "summary": self.summary, "dialogue": self.dialogue}

    def add_dialogue(self, user: str, text: str) -> None:
        self.dialogue += f"{user}: {text}\n"


class MessageThread:
    def __init__(self, id, messages: list):
        self.id = id
        self.messages = messages

    def to_json(self):
        return {"id": self.id, "messages": [message.to_json() for message in self.messages]}
