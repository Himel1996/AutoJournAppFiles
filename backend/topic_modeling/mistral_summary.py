from topic_modeling.topic_modeling import TopicModeling
from umap import UMAP
from sentence_transformers import SentenceTransformer
import re
import json
import numpy as np
import pandas as pd
from langchain_ollama.llms import OllamaLLM
# from langchain.llms import Ollama
#from langchain_community.llms import Ollama
import asyncio
from concurrent.futures import ThreadPoolExecutor

class MistralTopicModeling(TopicModeling):
    """
    Topic Modeling using the Mistral model via LangChain's OllamaLLM.
    Dynamically generates topics based on input data and a user-defined prompt.
    """

    def __init__(self, num_topics: int=5, model_name: str = "mistral", redis_host="localhost", redis_port=6379):
        self.num_topics = num_topics
        self.model_mistral = OllamaLLM(model=model_name)  # Initialize the Ollama LLM with the Mistral model.
        # self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")  # For embeddings.
        # self.umap_model = UMAP(n_neighbors=15,
        #                        transform_seed=173,  # Fix seed for reproducibility.
        #                        n_components=5,
        #                        min_dist=0.0,
        #                        metric='cosine')

    def preprocess(self, text):
        """Clean the input text by removing links, tags, and emojis."""
        text = re.sub(r"http\S+", "", text)  # Remove links.
        text = re.sub(r"@\S+", "", text)  # Remove tags.
        text = re.sub(r"\w+:\s?", "", text)  # Remove author at the start.
        text = self.__remove_emojis(text)

        text = re.sub(r"@\w+", "", text)  # Remove mentions
        text = re.sub(r"#\w+", "", text)  # Remove hashtags
        text = re.sub(r"[^A-Za-z0-9\s]", "", text)  # Remove non-alphanumeric characters
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
        return text

    def __remove_emojis(self, text):
        """Remove emojis from text."""
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # Emoticons.
                                   u"\U0001F300-\U0001F5FF"  # Symbols & pictographs.
                                   u"\U0001F680-\U0001F6FF"  # Transport & map symbols.
                                   u"\U0001F1E0-\U0001F1FF"  # Flags.
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    # Function to clean and extract JSON from a response
    def extract_json(self, response_text):
        try:
        # Use a regex to find the JSON object within the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
        return None
    def clean_response(self, response):
        response = response.replace("\n", "")
        cleaned_str = re.sub(r"```json|```", "", response)  # Remove backticks and "```json"
        cleaned_str = re.sub(r"\\", "", cleaned_str)  # Remove backslashes
        cleaned_str = re.sub(r"\'", '"', cleaned_str)  # Replace escaped single quotes (\' -> ")
        cleaned_str = re.sub(r'"s ', ' ', cleaned_str)  # Replace "s with space
        cleaned_str = re.sub(r"\s+", " ", cleaned_str)  # Replace multiple spaces with single space
        return cleaned_str.strip()
        
    def get_topics(self, doc):
        """
        Generate topics dynamically using the Mistral model.
        Returns topic probabilities, keywords, topic names, and topic details.
        """
        preprocessed_doc = self.preprocess(doc)
        prompt = f"""
        Analyze the following text and provide at least {self.num_topics} most suitable topics for it, along with their percentages and the relevant 15 keywords.
        Text:
        {preprocessed_doc}
        Format the output as a JSON object: {{"topics": [{{"name": "Topic1", "percentage": 25.0, "keywords": ['keyword1','keyword2',..]}}, ...]}}
        """
        try:
            response = self.model_mistral.generate([prompt])
            response_text = response.generations[0][0].text.strip()
            cleaned_response = self.clean_response(response_text)

            data = self.extract_json(cleaned_response)
            if not data:
                raise ValueError("No valid JSON extracted from model response.")

            topics_and_keywords = data["topics"]
            topics = [topic["name"] for topic in topics_and_keywords]
            probs = {topic["name"]: topic["percentage"] for topic in topics_and_keywords}
            keywords = {topic["name"]: topic["keywords"] for topic in topics_and_keywords}

            return probs, keywords, topics, topics_and_keywords
        except Exception as e:
            print(f"Error in get_topics: {e}")
            return None, None, None, None
        
    def get_topics_test(self, doc):
        """
        Mock version of the get_topics method for testing without GPU.
        Generates hardcoded topics and keywords as a response.
        
        Parameters:
        - doc: Input document (not used in this test version).
        
        Returns:
        - probs: Hardcoded probabilities for each topic.
        - keywords: Hardcoded keywords for each topic.
        - topics: List of topic names.
        """
        # Example hardcoded response
        probs = {
            "Topic 1": 30.0,
            "Topic 2": 25.0,
            "Topic 3": 20.0,
            "Topic 4": 15.0,
            "Topic 5": 10.0
        }
        
        keywords = {
            "Topic 1": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
            "Topic 2": ["keyword6", "keyword7", "keyword8", "keyword9", "keyword10"],
            "Topic 3": ["keyword11", "keyword12", "keyword13", "keyword14", "keyword15"],
            "Topic 4": ["keyword16", "keyword17", "keyword18", "keyword19", "keyword20"],
            "Topic 5": ["keyword21", "keyword22", "keyword23", "keyword24", "keyword25"]
        }
        
        topics = list(probs.keys())

        topics_and_keywords = [
            {"name": topic, "percentage": probs[topic], "keywords": keywords[topic]} for topic in topics
        ]
        
        return probs, keywords, topics, topics_and_keywords

    def check_topic_count(self, num_topics):
        """Ensure the number of topics matches the specified count."""
        if self.num_topics != num_topics:
            self.num_topics = num_topics
    

    ##Summarization Zone:
   
    def summarize_with_hint(self, text, topics_and_keywords):
        """Generate summaries for each topic using LLM."""
        summaries = []
        for topic in topics_and_keywords:
            topic_name = topic["name"]
            keywords = ', '.join(topic["keywords"][:5])  # Use top 5 keywords
            prompt = f"""
            Summarize the following text in 2-3 sentences, focusing on the provided topic and keywords.
            Hint: Topic - {topic_name}; Keywords - {keywords}.

            Ensure the summary is concise and captures the main points of the text without adding many details.
            If the text is very short, condense the summary into a single, clear, and well-structured sentence.
            Avoid repeating keywords unnecessarily or adding unrelated details.

            Text: {text}
            """
            try:
                response = self.model_mistral(prompt)
                if response.strip():
                    summaries.append({"topic": topic_name, "summary": response.strip()})
            except Exception as e:
                print(f"Error summarizing for topic {topic_name}: {e}")
        return summaries
    
    def summarize_with_hint_test(self, text, topics_and_keywords):
        """Mock version of summarize_with_hint for testing purposes."""
        summaries = []
        for topic in topics_and_keywords:
            topic_name = topic["name"]
            keywords = ', '.join(topic["keywords"][:5])  # Use top 5 keywords
            mock_summary = f"This is a mock summary for the topic '{topic_name}' with focus on keywords: {keywords}."
            summaries.append({"topic": topic_name, "summary": mock_summary})
        return summaries
    
    def generate_news_article(self, topic, summary, keywords, style):
        """Generate a long news article based on topic, summary, keywords, and style."""
        style_prompt = {
            "formal": "Use a professional and objective tone, suitable for a reputable news outlet.",
            "academic": "Write in an academic style, using analytical and precise language.",
            "gen_z": "Use a Gen Z tone, be witty, include pop culture references, and keep it conversational.",
            "narrative": "Write in a storytelling style, with vivid descriptions and engaging narrative techniques.",
            "persuasive": "Write persuasively, using emotional and motivational language.",
            "satirical": "Write in a satirical tone, with humor and irony to critique the subject.",
            "conversational": "Use a friendly and informal conversational tone.",
            "poetic": "Write in a poetic style, with metaphorical and rhythmic language.",
            "investigative": "Write in an investigative tone, presenting facts systematically and focusing on analysis."
        }

        tone_instructions = style_prompt[style]

        prompt = (
            f"You are a professional journalist writing for a major news outlet. Your goal is to craft a compelling and detailed news article.\n\n"
            f"**Topic**: {topic}\n\n"
            f"**Summary**: {summary}\n\n"
            f"**Key Points and Keywords**:\n- " + "\n- ".join(keywords) + "\n\n"
            f"**Style**: {tone_instructions}\n\n"
            f"**Requirements**:\n"
            f"1. Write a long, engaging news article (at least 800 words).\n"
            f"2. Include an attention-grabbing headline at the beginning.\n"
            f"3. Expand upon the provided summary using the listed keywords. Use them naturally throughout the article.\n"
            f"4. Include historical context, background, or analysis where relevant.\n"
            f"5. Use the specified style and tone throughout the article.\n\n"
            f"Start your response with the headline, followed by the full article."
        )

        print(f"Generating article for topic: {topic} in {style} style")
        response = self.model_mistral(prompt)
        print(f"Article generated for topic: {topic}")
        return response
    
    def generate_news_article_test(self, topic, summary, keywords, style):
        """Mock version of generate_news_article for testing purposes."""
        style_descriptions = {
            "formal": "a professional and objective tone, suitable for a reputable news outlet.",
            "academic": "an academic style, using analytical and precise language.",
            "gen_z": "a Gen Z tone, witty and conversational with pop culture references.",
            "narrative": "a storytelling style with vivid descriptions.",
            "persuasive": "a persuasive tone, using emotional and motivational language.",
            "satirical": "a satirical tone with humor and irony.",
            "conversational": "a friendly and informal conversational tone.",
            "poetic": "a poetic style, using metaphorical and rhythmic language.",
            "investigative": "an investigative tone, presenting facts systematically and focusing on analysis."
        }

        style_description = style_descriptions.get(style, "a general style")

        article = f"""
        **Headline**: Breaking News: {topic}

        In a remarkable development, {summary}

        This article explores the topic of "{topic}" with a focus on the following key points:
        - {', '.join(keywords)}

        Written in {style_description}, this piece dives into the intricacies of {topic}, providing insights and detailed analysis.
        """
        return article.strip()

