from telethon import TelegramClient
from telethon import functions, types
import asyncio
from summarization.models.samsum import Samsum, MessageThread
import config as config
from api_connection.social_api import SocialAPI


class TelegramAPI(SocialAPI):
    def __init__(self):
        self.client = TelegramClient(
            'nlp_user', config.Config.TELEGRAM_API_ID, config.Config.TELEGRAM_API_HASH)

    async def get_conversations(self, query: str, channel_limit=5, message_limit=10, parse_func=None):
        result = await self.query_async(query, channel_limit, message_limit)
        return parse_func(result) if parse_func else self.parse_all_messages_json(result)

    def query(self, query: str, channel_limit=5, message_limit=10) -> list[MessageThread]:
        result = asyncio.run(self.query_async(
            query, channel_limit, message_limit))
        return result

    async def query_async(self, query: str, channel_limit=5, message_limit=10) -> list[MessageThread]:
        async with self.client:
            await self.start_app()
            channels = await self.search_channels(query, channel_limit)
            query_result = []

            for channel in channels:
                try:
                    messages = await self.get_messages_from_channel(channel.title, message_limit)
                    message_thread = MessageThread(channel.id, messages)
                    query_result.append(message_thread)
                except Exception as e:
                    print(e)
                    continue

            return query_result

    async def start_app(self) -> None:
        await self.client.start()

        passkey = config.Config.TELEGRAM_PASSWORD
        phone = config.Config.TELEGRAM_PHONE_NUMBER

        if not await self.client.is_user_authorized():
            await self.client.send_code_request(phone)
        try:
            await self.client.sign_in(phone, passkey)
        except Exception:
            await self.client.sign_in(password=input('Password: '))

    async def get_messages_from_channel(self, channel_name: str, limit=10) -> list[types.Message]:
        messages = []
        async for message in self.client.iter_messages(channel_name, limit=limit):
            messages.append(message)

        return messages

    async def search_channels(self, search_word: str, limit=5) -> list[types.Channel]:
        channels = []

        result = await self.client(functions.contacts.SearchRequest(
            q=search_word,
            limit=limit
        ))

        for chat in result.chats:
            if isinstance(chat, types.Channel):
                channels.append(chat)

        return channels

    def parse_message(self, messages: list[types.Message], id) -> Samsum:
        samsum = Samsum(id, "", "")
        for message in messages:
            try:
                samsum.add_dialogue(message.sender_id, message.text)
            except Exception as e:
                print(e)

        return samsum

    def parse_message_json(self, messages: list[types.Message], id) -> dict:
        return self.parse_message(messages, id).to_json()
