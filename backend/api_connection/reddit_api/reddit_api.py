import praw
import config as config
from summarization.models.samsum import Samsum, MessageThread
from api_connection.social_api import SocialAPI
from praw import models


class RedditAPI(SocialAPI):
    base_url = "https://www.reddit.com/"
    oauth_base_url = "https://oauth.reddit.com/"

    def __init__(self):
        self.client = praw.Reddit(
            client_id=config.Config.REDDIT_API_ID,
            client_secret=config.Config.REDDIT_API_SECRET,
            user_agent=config.Config.REDDIT_APP_NAME,
            username=config.Config.REDDIT_USERNAME,
            password=config.Config.REDDIT_PASSWORD
        )

    def get_conversations(self, query: str, limit=5, parse_func=None) -> Samsum:
        result = self.search(query, limit)
        return parse_func(result) if parse_func else self.parse_all_messages_json(result)

    def search(self, query: str, limit=5) -> list[MessageThread]:
        reddit = self.client.subreddit("all")

        # parse query for subreddit, there should be no space. Replace space with ''
        subreddit_name = query.replace(" ", "")

        subreddit = self.client.subreddit(subreddit_name)

        resultsAll = reddit.search(query, limit=limit)
        resultSubreddit = subreddit.search(query, limit=limit)

        results = list(resultsAll) + list(resultSubreddit)

        return self.get_message_threads(results)

    def get_message_threads(self, results) -> list[MessageThread]:
        message_threads = []
        for submission in results:
            comments = self.extract_comments(submission)

            # the post title as the first comment of the post
            first_comment = models.Comment(submission, submission.title)
            comments.insert(0, first_comment)
            message_thread = MessageThread(submission.id, comments)
            message_threads.append(message_thread)

        return message_threads

    def parse_results(self, results) -> list[MessageThread]:
        message_threads = []
        for submission in results:
            message_thread = MessageThread(
                submission.author, submission.title)
            message_threads.append(message_thread)

        return message_threads

    def extract_comments(self, submission: models.Submission, comment_limits=15) -> list:
        return submission.comments.list()[:comment_limits]

    def parse_message_json(self, messages: list, id: str) -> dict:
        return self.parse_message(messages, id).to_json()

    def parse_message(self, messages: list, id: str) -> Samsum:
        dialogue = ""
        for message in messages:
            try:
                dialogue += f"{message.author}: {message.body}\n"
            except Exception:
                dialogue += f"{message._reddit.author}: {message._reddit.title}\n"

        return Samsum(id, "", dialogue)
