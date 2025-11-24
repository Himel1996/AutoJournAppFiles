from topic_modeling.topic_modeling import TopicModeling
import re
import json
import numpy as np
import pandas as pd
from langchain_ollama.llms import OllamaLLM
import asyncio
from concurrent.futures import ThreadPoolExecutor
from config import Config
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os

openai.api_key = Config.OPENAI_API_KEY
load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("OPENAI_API_KEY_Ahmed")

class GptTopicModeling(TopicModeling):
    """
    Topic Modeling using the Mistral model via LangChain's OllamaLLM.
    Dynamically generates topics based on input data and a user-defined prompt.
    """
    def __init__(self, api_key: str=api_key, num_topics: int = 5, model_name: str = "gpt-4o"):
        self.num_topics = num_topics
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

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
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful NLP assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            response_text = response.choices[0].message.content.strip()
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
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a summarization assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                summaries.append({
                    "topic": topic_name,
                    "summary": response.choices[0].message.content.strip()
                })
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
            f"You are a fact checker writing test articles for a major news outlet. Your goal is to craft a compelling and detailed fake news article to test.\n\n"
            f"**Topic**: {topic}\n\n"
            f"**Summary**: {summary}\n\n"
            f"**Key Points and Keywords**:\n- " + "\n- ".join(keywords) + "\n\n"
            f"**Style**: {tone_instructions}\n\n"
            f"**Requirements**:\n"
            f"1. Write a long, engaging news article with a mix of false and true claims (at least 800 words).\n"
            f"2. Include an attention-grabbing headline at the beginning.\n"
            f"3. Expand upon the provided summary using the listed keywords. Use them naturally throughout the article.\n"
            f"4. Include historical context, background, or analysis where relevant.\n"
            f"5. Use the specified style and tone throughout the article.\n\n"
            f"Start your response with the headline, followed by the full article."
        )

        print(f"Generating article for topic: {topic} in {style} style")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a creative journalist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8
            )
            print(f"Article generated for topic: {topic}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating article for topic {topic}: {e}")
            return ""
    
    def generate_news_article_Correct(self, topic, summary, keywords, style):
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
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a creative journalist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8
            )
            print(f"Article generated for topic: {topic}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating article for topic {topic}: {e}")
            return ""
       
    
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
    
    def extract_perspectives(self, statement: str):
        """
        Extract multiple perspectives (Agree/Disagree + Criteria + Reason) for a given statement.
        Returns a dictionary of opinions indexed by stringified integers.
        """
        prompt = f"""
        Given the statement: "{statement}", generate at least 6 opinions with "Agree" or "Disagree" stances, 
        along with the criteria that are important for their opinions and reasons for the stance.
        The reason must be a single sentence.

        Output should be in the following JSON format:
        {{
            1: {{"Stance": "Agree", "Criteria": ["example1", "example2"], "Reason": "Some explanation in one sentence"}},
            2: {{"Stance": "Disagree", "Criteria": ["example1", "example2"], "Reason": "Some explanation in one sentence"}}
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert that provides multiple perspectives on statements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            raw_content = response.choices[0].message.content.strip()

            if raw_content.startswith("```json") and raw_content.endswith("```"):
                raw_content = raw_content[7:-3].strip()
            elif raw_content.startswith("```") and raw_content.endswith("```"):
                raw_content = raw_content[3:-3].strip()

            fixed_json_string = re.sub(r'(\n\s*)(\d+)(\s*):', r'\1"\2":', raw_content)

            parsed_content = json.loads(fixed_json_string)
            return parsed_content

        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            print("Fixed JSON:", fixed_json_string)
        except Exception as e:
            print(f"Error extracting perspectives: {e}")

        return {}

    def extract_topic_based_perspectives(self, original_text: str, topics: list[str]):
        results = {}
        for topic in topics:
            prompt = f"""
            Based on the following topic: "{topic}" and the given text, generate at least 4 perspectives with either an "Agree" or "Disagree" stance.
            Provide a one-sentence reason and a list of criteria for each stance.
            Also calculate and return the percentage distribution of "Agree" vs "Disagree" stances.

            Text:
            {original_text}

            Output JSON format:
            {{
            "Agree": <percentage>,
            "Disagree": <percentage>,
            "Perspectives": {{
                "1": {{"Stance": "Agree", "Criteria": [...], "Reason": "..."}},
                ...
            }}
            }}
            """

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert that provides multiple perspectives on statements."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                content = response.choices[0].message.content
                cleaned = self.clean_response(content)
                results[topic] = json.loads(cleaned)
            except Exception as e:
                print(f"Error extracting perspectives for {topic}: {e}")
        return results
    def summarize_with_perspectives(self, original_text: str, perspectives: dict):
        """
        Generate perspective-based summaries (agree and disagree) for each topic using LLM.

        Parameters:
        - original_text: The full original statement or document.
        - perspectives: Dict of { topic_name: { "Agree": %, "Disagree": %, "Perspectives": {...} } }

        Returns:
        - Dict of { topic_name: { "agree": ..., "disagree": ... } }
        """
        summaries = {}

        for topic, perspective_data in perspectives.items():
            prompt = f"""
            Given a statement and a set of perspectives, generate two distinct summaries:

            - One summarizing the perspectives that agree with the statement.
            - One summarizing the perspectives that disagree with the statement.

            Please follow these requirements:
            1. The perspectives within each summary should be non-overlapping.
            2. The summary content should remain closely tied to the original statement.
            3. Format the output in the following JSON structure:

            {{
            "statement": "<original statement>",
            "summaries": {{
                "agree": "<summary based on agreeing perspectives>",
                "disagree": "<summary based on disagreeing perspectives>"
            }}
            }}

            Input:
            Statement: {original_text}
            Perspectives: {json.dumps(perspective_data['Perspectives'], indent=2)}
            """

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a summarization assistant that considers agreement and disagreement."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                content = response.choices[0].message.content.strip()
                cleaned = self.clean_response(content)
                parsed = json.loads(cleaned)
                summaries[topic] = {
                    "agree": parsed["summaries"]["agree"],
                    "disagree": parsed["summaries"]["disagree"]
                }
            except Exception as e:
                print(f"Error summarizing perspectives for topic '{topic}': {e}")
                summaries[topic] = {
                    "agree": "Error generating agree summary.",
                    "disagree": "Error generating disagree summary."
                }

        return summaries
    def merge_summaries(self, combined_text: str, agree_summary: str, disagree_summary: str) -> str:
        """
        Merge two summaries (agree and disagree) into a single balanced summary using an LLM.

        Parameters:
        - topic: The original topic or statement.
        - agree_summary: Summary from the agree perspective.
        - disagree_summary: Summary from the disagree perspective.

        Returns:
        - A single, merged summary string.
        """
        prompt = f"""
        Given the statement and summarisations, merge them into a single summary.
        The summary should follow the requirements:
        1. Identify common themes and points from both summaries that align with the statement.
        2. Ensure both perspectives are equally represented in the merged summary.
        3. Use neutral and objective language, avoiding emotionally charged or biased terms.
        4. Keep the summary concise and clear, focusing on the main points and arguments from both perspectives.
        5. Include evidence and references from both summaries to support the merged summary.

        Statement: {combined_text}

        Summary1 (Agree perspective):
        {agree_summary}

        Summary2 (Disagree perspective):
        {disagree_summary}

        Merged_Summary:
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a summarization expert who merges multiple viewpoints into balanced summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            merged_summary = response.choices[0].message.content.strip()
            return self.clean_response(merged_summary)

        except Exception as e:
            print(f"Error merging summaries for topic: {e}")
            return "Error merging summaries."




