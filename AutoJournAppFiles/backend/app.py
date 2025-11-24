"""
    NLP Summarization & Topic Modeling.
    Current implemented endpoints:
    - "/search": fetch Twitter conversations related to passed query.
        - GET request
        - Query Parameters: query: <String>
        - Response is a JSON object of list of conversations.

    - "/topics": Perform topic modeling on passed twitter conversations.
        - GET request
        - Query Parameters: num_topics: <Int>
        - Body must be json of the format:
            {
                "conversation": <list of twitter conversation as returned from the /search endpoint>,
            }
        - Response is a JSON Object of conversations topics probabilities and topic names.

   - "/summarize": Perform summarization on passed twitter conversations.
        - GET request
        - Body must be json of the format:
            {
                "conversation": <list of twitter conversation as returned from the /search endpoint>,
            }
        - Response is a JSON Object of conversations summaries.

    - "/topic-aware-summarize": Perform topic aware summarization on passed twitter conversations.
        - POST request
        - Body must be json of the format:
            {
                "conversation": <list of twitter conversation as returned from the /search endpoint>,
            }
        - Response is a JSON Object of conversations summaries with respect to different topics.

    - "/delta-summarize": Perform delta summarization on passed topic aware summaries.
        - POST request
        - Body must be json of the format:
            {
                "summaries": <dict of topic aware summaries>,
            }
        - Response is a type of plot (image) showing how different are the summarizations.
"""

import json
from language_detection.lang_detection import LangDetect
# from summarization.delta_summarization.delta_summarization import DeltaSummarization
# from summarization.models.bart import Bart
# from summarization.agents.agent_factory import AgentsFactory
# from summarization.topic_aware_summarization.topic_aware_summarization import TopicAwareSummarization
# from api_connection.twitter_api.twitter_api import TweetAPI
from api_connection.telegram_api.telegram_api import TelegramAPI
from api_connection.reddit_api.reddit_api import RedditAPI
from topic_modeling.Bertopic import Bertopic
#from topic_modeling.mistral import MistralTopicModeling
from topic_modeling.mistral_summary import MistralTopicModeling
from topic_modeling.gpt import GptTopicModeling
from asgiref.wsgi import WsgiToAsgi
from flask import Flask, request, jsonify, send_file, redirect, url_for
from flask_restx import Api, Resource, fields
import config
import logging
import asyncio
from redis.asyncio import Redis
from nltk.tokenize import sent_tokenize
import re

import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import networkx as nx
from networkx.algorithms import community
from collections import defaultdict
from keybert import KeyBERT
from rapidfuzz import process, fuzz

perspective_cache = {}
conversation_cache = None
summary_cache = {}
merged_summary_cache = {}
article_cache= {}

#Bias detection models
bias_detector = pipeline("text-classification", model="himel7/bias-detector", tokenizer="roberta-base")
bias_type_classifier = pipeline("text-classification", model="maximuspowers/bias-type-classifier")
neutralizer_model_name = "himel7/bias-neutralizer-t5s"
neutralizer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
neutralizer_tokenizer = AutoTokenizer.from_pretrained(neutralizer_model_name)
neutralizer_model = AutoModelForSeq2SeqLM.from_pretrained(neutralizer_model_name)
neutralizer_model.to(neutralizer_device)

def neutralize_bias(sentence):
    input_text = "neutralize: " + sentence
    inputs = neutralizer_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(neutralizer_device)
    output_ids = neutralizer_model.generate(**inputs, max_length=128, num_beams=4)
    sent=neutralizer_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return sent.capitalize()

from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("OPENAI_API_KEY_Ahmed")
google_api=os.getenv("Google_API")
serp_api=os.getenv("serp_api")
cx=os.getenv("cx")
client = OpenAI(api_key=api_key)
model_name = "gpt-4o"
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
DEFAULT_LANGUAGE = "en"

# Function to clean and extract JSON from a response
def extract_json(response_text):
    try:
    # Use a regex to find the JSON object within the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
    return None
def clean_response(response):
    response = response.replace("\n", "")
    cleaned_str = re.sub(r"```json|```", "", response)  # Remove backticks and "```json"
    cleaned_str = re.sub(r"\\", "", cleaned_str)  # Remove backslashes
    cleaned_str = re.sub(r"\'", '"', cleaned_str)  # Replace escaped single quotes (\' -> ")
    cleaned_str = re.sub(r'"s ', ' ', cleaned_str)  # Replace "s with space
    cleaned_str = re.sub(r"\s+", " ", cleaned_str)  # Replace multiple spaces with single space
    return cleaned_str.strip()

def neutralize_with_gpt(sentence):
    prompt = (
        f"The following sentence contains biased language:\n\n"
        f"{sentence}\n\n"
        f"Rewrite it in a neutral and objective tone."
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()

                                                      
# Initialize the application's components
app = Flask('NLPLAB')

# Initialize Redis client globally
redis_client = Redis.from_url("redis://localhost:6379", decode_responses=True)

api = Api(app, version='1.0',
          title='Automated Journalist App',
          description='API for Automated Journalist App - NLP Summarization & Topic Modeling')

asgi_app = WsgiToAsgi(app)
config.init()

# twitter_api = TweetAPI()
tele_api = TelegramAPI()
reddit_api = RedditAPI()

# bertopic = Bertopic(num_topics=10)  # Default number of topics is 10.
#mistral_modeling= MistralTopicModeling(model_name='mistral')
mistral_modeling= GptTopicModeling(model_name='gpt-4o')
# summarizer_model = Bart(config.Config.SUMMARIZATION_MODEL)
# summarizer_agent = AgentsFactory.get_agent(summarizer_model)
# topic_aware_summarizer = TopicAwareSummarization()
# delta_summarizer = DeltaSummarization()
lang_detect = LangDetect()

samsum = api.model('Samsum', {
    'id': fields.String(description='id of the conversation'),
    'summary': fields.String(description='summary of the conversation'),
    'dialogue': fields.String(description='dialogue to summarize'),
})

if __name__ != '__main__':
    # App is being run externally (through gunicorn).
    gunicorn_logger_access = logging.getLogger("gunicorn.access")
    # Use the gunicorn logger as the app logger.
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    # Use the specified log level.
    app.logger.setLevel(gunicorn_logger.level)


@app.route('/search-telegram', methods=["GET"])
async def fetch_telegram():
    query = request.args["query"]
    response = await tele_api.get_conversations(query, channel_limit=config.Config.MAX_NUM_OF_TELEGRAM_CHANNELS,
                                                message_limit=config.Config.MAX_NUM_OF_TELEGRAM_MESSAGES_PER_CHANNEL)
    final_response = []
    for conv in response:
        try:
            if lang_detect.is_english(conv['dialogue']):
                final_response.append(conv)
        except:
            continue

    return jsonify({"conversations": final_response})


@api.route('/search-reddit')
class SearchReddit(Resource):
    @api.doc(params={'query': 'a string'})
    def get(self):
        query = request.args["query"]
        response = reddit_api.get_conversations(query, limit=5)
        return jsonify({"conversations": response})


# @app.route('/search-twitter', methods=["GET"])
# def fetch_tweets():
#     query = request.args["query"]
#     response = twitter_api.get_conversations(search_keyword=query,
#                                      max_num_conv=config.Config.API_MAX_NUM_CONVERSATIONS,
#                                      max_num_pages=config.Config.API_MAX_NUM_PAGES,
#                                      max_page_res=config.Config.API_MAX_PAGE_NUM_RESULTS,
#                                      parse_func=twitter_api.parse_as_conv_hierarchy)
#     return jsonify({"conversations": response})


# @api.route('/topics')
# class Topics(Resource):
#     @api.doc(body=api.model(
#         'Topics',
#         {
#             'conversations': fields.List(fields.String, description='list of conversations'),
#             'num_topics': fields.Integer(description='number of topics to extract')
#         })
#     )
#     def post(self):
#         conversation_list = request.json["conversations"]
#         num_topics = int(request.json["num_topics"])
#         # Update topic count if necessary.
#         bertopic.check_topic_count(num_topics)
#         if config.Config.TOPIC_PER_TWEET:
#             conv_topic_probs, topics = bertopic.run_tweet_topic_modeling(
#                 conversation_list)
#         else:
#             conv_topic_probs, topics = bertopic.run_con_topic_modeling(
#                 conversation_list)

#         return jsonify({"topics": conv_topic_probs, "index_to_topic": topics})

from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=5)
@api.route('/topics/Mistral', methods=['POST'])
class Topics(Resource):

    def post(self):
        selectedOption= "mistral"

        try:
            # Parse input
            conversation_list = request.json["conversations"]
            combined_conversations = " ".join(conversation_list)

            global conversation_cache
            conversation_cache = combined_conversations

            # Dynamically choose the topic modeling method
            if selectedOption.lower() == "mistral":
                # probs, keywords, topics, topics_and_keywords = mistral_modeling.get_topics_test(combined_conversations)
                probs, keywords, topics, topics_and_keywords = mistral_modeling.get_topics(combined_conversations)
            else:
                return {"error": f"Invalid option: {selectedOption}"}, 400
            
            def extract_and_cache():
                global perspective_cache
                perspective_cache = mistral_modeling.extract_topic_based_perspectives(combined_conversations, topics)
            executor.submit(extract_and_cache)
            print("Perspectives Cached")
            print(perspective_cache)

            # Create a response
            response = {
                "topics": probs,  # Probabilities for each topic
                "index_to_topic": topics,  # List of topic names
                "keywords": keywords,  # Keywords associated with each topic
                "topics_and_keywords": topics_and_keywords
            }
            return jsonify(response)
        except Exception as e:
            return {"error": str(e)}, 400
#Route for Perspective
@api.route("/topics/Mistral/perspectives", methods=["GET"])
class TopicPerspectives(Resource):
    def get(self):
        if perspective_cache:
            return jsonify({"perspectives": perspective_cache})
        return {"error": "No perspectives found."}, 404
    
#Route for Summary:
@api.route('/summarize-perspectives')
class SummarizePerspectives(Resource):
    def post(self):
        try:
            data = request.json
            topic = data.get("topic")
            #print(f"[Request] Received topic: {topic}")
            if topic in summary_cache:
                return jsonify(summary_cache[topic])
            
            # Check if required global state is available
            if conversation_cache is None:
                print("[Error] conversation_cache is None")
                return {"error": "No conversation data available."}, 400

            if topic not in perspective_cache:
                return {"error": f"No perspectives found for topic '{topic}'"}, 404
            print(perspective_cache[topic])
            print("calling summarize_with_perspectives")
            print("calling summarize_with_perspectives")
            try:
                all_summaries = mistral_modeling.summarize_with_perspectives(
                    conversation_cache, {topic: perspective_cache[topic]}
                )
                #print(f"[Summarize Output Keys] {list(all_summaries.keys())}")
                summary = all_summaries[topic]
                #print(f"[Success] Summary: {summary}")
            except Exception as e:
                print(f"[Error] summarize_with_perspectives failed: {e}")
                return {"error": str(e)}, 400
            
            """ all_summaries = mistral_modeling.summarize_with_perspectives(
                conversation_cache, {topic: perspective_cache[topic]}
            ) """
            summary = all_summaries[topic]
            #print(f"[Success] Summary: {summary}")
            
            summary_cache[topic] = all_summaries[topic]
            return jsonify(summary)
        except Exception as e:
            return {"error": str(e)}, 400


#Route for merge    
@api.route('/merge-summaries')
class MergeSummaries(Resource):
    def post(self):
        try:
            topic = request.json["topic"]
            agree_summary = request.json["agree"]
            disagree_summary = request.json["disagree"]

            if topic in merged_summary_cache:
                return jsonify({"merged": merged_summary_cache[topic]})

            merged = mistral_modeling.merge_summaries(conversation_cache, agree_summary, disagree_summary)
            merged_summary_cache[topic] = merged
            return jsonify({"merged": merged})
        except Exception as e:
            return {"error": str(e)}, 400
        
def custom_split_article(article):
    lines = article.split('\n')
    result = []
    buffer = ""

    for line in lines:
        stripped = line.strip()

        # Detect and preserve markdown headers
        if re.match(r'^#+\s', stripped) or re.match(r'^\*\*.*\*\*$', stripped):
            # First flush the buffer as normal text
            if buffer:
                result.extend(sent_tokenize(buffer.strip()))
                buffer = ""

            # Preserve heading with double newline
            result.append(stripped + "\n\n")
            continue

        # Collect regular content
        if stripped:
            buffer += ' ' + stripped

    # Final flush of remaining buffered text
    if buffer:
        result.extend(sent_tokenize(buffer.strip()))

    return result
#Route for news
@api.route('/generate-news', methods=['POST'])
class GenerateNews(Resource):
    def post(self):
        try:
            topic = request.json.get("topic")
            keywords = request.json.get("keywordsSent")
            style = request.json.get("style")

            if topic not in merged_summary_cache:
                return {"error": "No merged summary found."}, 404

            article = mistral_modeling.generate_news_article(
                topic, merged_summary_cache[topic], keywords, style
            )

            allsentences = custom_split_article(article)
            sentences = custom_split_article(article)
            # Cache full article and sentences
            global article_cache
            article_cache["sentences"] = sentences
            article_cache["article"] = article
            article_cache["allsentences"] = allsentences


            return jsonify({"article": article})
        except Exception as e:
            return {"error": str(e)}, 400
        
#Bias detection endpoint
@api.route('/detect-bias', methods=['POST'])
class DetectBias(Resource):
    def post(self):
        try:
            if "sentences" not in article_cache:
                return {"error": "No article sentences cached."}, 400

            sentences = article_cache["sentences"]
            bias_flags = []
            biased_sentences = []
            scores = []
            allsentences= article_cache["allsentences"]

            for sentence in sentences:
                result = bias_detector(sentence)[0]
                label = result["label"]
                score = result["score"]
                is_biased = label == "LABEL_1"
                bias_flags.append({
                    "sentence": sentence,
                    "biased": is_biased,
                    "score": round(score, 4)
                })
                if is_biased:
                    biased_sentences.append(sentence)
                    scores.append(round(score, 4))

            article_cache["bias_flags"] = bias_flags
            return jsonify({
                "bias_flags": bias_flags,
                "biased_sentences": biased_sentences,
                "scores": scores,
                "all_sentences": allsentences
            })
        except Exception as e:
            return {"error": str(e)}, 500
        
#Bias type classifier endpoint
@api.route('/bias-type', methods=['POST'])
class BiasType(Resource):
    def post(self):
        try:
            if "bias_flags" not in article_cache:
                return {"error": "No bias flags found. Run /detect-bias first."}, 400

            biased_sentences = [x["sentence"] for x in article_cache["bias_flags"] if x["biased"]]
            bias_types = []

            for sentence in biased_sentences:
                result = bias_type_classifier(sentence)[0]
                bias_types.append({
                    "sentence": sentence,
                    "bias_type": result["label"],
                    "score": round(result["score"], 4)
                })

            article_cache["bias_types"] = bias_types
            return jsonify({"bias_types": bias_types})
        except Exception as e:
            return {"error": str(e)}, 500


#Bias neutralize
@api.route('/neutralize-bias')
class NeutralizeBias(Resource):
    def post(self):
        try:
            data = request.json
            sentences = data.get("sentences", [])
            if not isinstance(sentences, list):
                return {"error": "Expected a list of sentences."}, 400
            
            neutralized_sentences = []
            for sentence in sentences:
                neutral = neutralize_bias(sentence)
                if neutral.strip().lower() == sentence.strip().lower():
                    print(f"[Fallback] Using GPT for: {sentence}")
                    neutral = neutralize_with_gpt(sentence)
                neutralized_sentences.append(neutral)

            #neutralized = [neutralize_bias(s) for s in sentences]
            return jsonify({"neutralized_sentences": neutralized_sentences})
        except Exception as e:
            return {"error": str(e)}, 500


#Community Detection

@api.route('/echo-chamber')
class EchoChamberDetection(Resource):
    def post(self):
        try:
            data = request.json
            original_text = data.get("originalText")
            topics = data.get("allTopics", {})

            if not original_text:
                return {"error": "Missing originalText"}, 400

            # Step 1: Parse user-comment mapping
            user_comments = extract_user_comments(original_text)

            # Step 2: Build a basic adjacency graph
            G = nx.Graph()
            users = list(user_comments.keys())
            for i in range(len(users) - 1):
                u1, u2 = users[i], users[i + 1]
                G.add_edge(u1, u2)

            # Step 3: Run sentiment analysis on each user's text
            sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", top_k=None)
            user_sentiments = {}
            for user, text in user_comments.items():
                s = sentiment(text[:512])[0]
                label = s[0]["label"]
                score = s[0]["score"]
                # Normalize: positive = 1, neutral = 0, negative = -1
                user_sentiments[user] = {
                    "positive": score,
                    "neutral": 0.0,
                    "negative": -score
                }.get(label, 0.0)

            # Step 4: Update edge weights by sentiment similarity
            for u, v in G.edges():
                sim = 1 - abs(user_sentiments.get(u, 0) - user_sentiments.get(v, 0))
                G[u][v]["weight"] = round(sim, 3)

            # Step 5: Community detection
            partition = community.louvain_communities(G, weight="weight", seed=42)
            # Convert to the same format as python-louvain (node: community_id dict)
            partition_dict = {}
            for i, community_set in enumerate(partition):
                for node in community_set:
                    partition_dict[node] = i
            print(f"Found {len(partition)} communities")
            print(f"Community sizes: {[len(comm) for comm in partition]}")

            #community keywords
            community_texts = defaultdict(str)
            for user, comm_id in partition_dict.items():
                community_texts[comm_id] += user_comments[user] + " "

            kw_model = KeyBERT()

            community_keywords = {}
            for comm_id, text in community_texts.items():
                keywords = kw_model.extract_keywords(text, top_n=5)
                community_keywords[comm_id] = [kw[0] for kw in keywords]

            # Step 6: Format output
            nodes = [
                {"id": node, "community": partition_dict[node], "label": community_keywords[partition_dict[node]][0], "sentiment": float(user_sentiments.get(node, 0.0))}
                for node in G.nodes()
            ]
            links = [{"source": u, "target": v, "value": G[u][v]["weight"]} for u, v in G.edges()]
            
            return jsonify({"nodes": nodes, "links": links})

        except Exception as e:
            return {"error": str(e)}, 500
        
def extract_user_comments(text):
    """Parses lines like 'Username: comment' into a dict."""
    lines = text.split("\n")
    pattern = re.compile(r"^([^\s:]{2,40}):\s*(.*)")
    comments = defaultdict(str)
    current_user = None

    for line in lines:
        match = pattern.match(line.strip())
        if match:
            current_user = match.group(1)
            comments[current_user] += match.group(2) + " "
        elif current_user:
            comments[current_user] += line.strip() + " "
    return dict(comments)

#echo-chamber detection

def detect_echo_extremes(text):
    prompt = (
        "Analyze the following text and identify pairs of opposing echo chambers. "
        "Return 2 to 3 sets of extremes based on controversial topics found in the text. "
        "Limit the extremes to 2/3 words only."
        "For each pair, format them like this:\n"
        "[\n"
        "  {\"extreme_1\": \"Pro-Trump\", \"extreme_2\": \"Anti-Trump\"},\n"
        "  {\"extreme_1\": \"Pro-Immigrants\", \"extreme_2\": \"Anti-Immigrants\"}\n"
        "]\n\n"
        f"Text:\n{text}"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    result = response.choices[0].message.content.strip()
    return result

@api.route('/detect-echo-extremes00')
class EchoExtremes(Resource):
    def post(self):
        try:
            data = request.json
            text = data.get("originalText", "")
            if not text:
                return {"error": "Missing originalText"}, 400

            result = detect_echo_extremes(text)

            # Try parsing as JSON
            try:
                parsed = json.loads(result)
                return jsonify({"extremes": parsed})
            except Exception:
                return jsonify({"raw": result, "error": "Could not parse JSON"}), 200

        except Exception as e:
            return {"error": str(e)}, 500
        

import json
@api.route('/detect-echo-extremes')
class DetectEchoExtremes(Resource):
    def post(self):
        try:
            data = request.json
            original_text = data.get("originalText", "")
            if not original_text.strip():
                return {"error": "originalText is empty."}, 400

            prompt = (
                f"The following text may represent polarized opinions or echo chambers:\n\n"
                f"{original_text}\n\n"
                "Identify a set of two opposing stance extremes found in this conversation or narrative. "
                "Return them as a JSON list with a 'stance_pair' and a one-line 'description' for each.\n\n"
                "Example:\n[\n"
                "  {\"stance_pair\": [\"Pro Trump\", \"Anti Trump\"], \"description\": \"Opinions regarding Trump's actions and policies.\"},\n"
                "  {\"stance_pair\": [\"Support Free Trade\", \"Protect Domestic Industry\"], \"description\": \"Perspectives on international trade policies.\"}\n"
                "]"
            )

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )

            output = response.choices[0].message.content.strip()
            cleaned_output = clean_response(output)
            # ✅ Remove ```json ... ``` block if present
            if output.startswith("```json"):
                output = re.sub(r"```json\s*([\s\S]+?)\s*```", r"\1", output).strip()

            try:
                extremes = json.loads(cleaned_output)
            except Exception as json_err:
                print(f"[DEBUG] Raw LLM output:\n{cleaned_output}")
                return {"error": f"Failed to parse LLM output as JSON: {str(json_err)}"}, 500

            return {"extremes": extremes}

        except Exception as e:
            print(f"[ERROR] Exception in /detect-echo-extremes:\n{str(e)}")
            return {"error": str(e)}, 500
        
@api.route('/shift-stance')
class ShiftStance(Resource):
    def post(self):
        try:
            data = request.json
            original_article = data.get("original_article")
            stance_pair = data.get("stance_pair")
            slider_value = int(data.get("slider_value", 50))

            if not original_article or not stance_pair:
                return {"error": "Missing data."}, 400

            # Determine bias tone
            strength = abs(slider_value - 50)
            direction = "toward " + (stance_pair[0] if slider_value > 50 else stance_pair[1])

            prompt = (
                f"Rewrite the following article with a tone slightly shifted {direction}. "
                f"The shift intensity is {strength}/50 (0 = neutral, 50 = extreme).\n\n"
                f"Original Article:\n{original_article}"
            )

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )

            return jsonify({"shifted_article": response.choices[0].message.content.strip()})
        except Exception as e:
            return {"error": str(e)}, 500
        
############# Claim Verification Zone Starts Here ###############################################

import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

def segment_article_into_sentences(article):
    return sent_tokenize(article)

def classify_sentences_as_claims(sentences, model="gpt-4", temperature=0.3):
    prompt = "Below is a list of sentences. For each one, answer YES if it is a factual claim that can be verified (true or false), and NO otherwise.\n\n"
    for i, sentence in enumerate(sentences):
        prompt += f"{i+1}. {sentence.strip()}\n"

    prompt += "\nAnswer as: 1. YES, 2. NO, ..."

    response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )

    #output = response['choices'][0]['message']['content']
    output = response.choices[0].message.content.strip()
    print("Out: ",output)
    #lines = output.strip().split("\n")
    lines = output.replace(",", "\n").split("\n")
    claim_sentences = []
    for i, line in enumerate(lines):
        if "YES" in line.upper():
            claim_sentences.append(sentences[i])
    return claim_sentences

## Extract Claims endpoint
@app.route("/extract-claims", methods=["POST"])
def extract_claims():
    try:
        data = request.json
        article = data.get("article", "")
        if not article.strip():
            return jsonify({"error": "Missing or empty article"}), 400

        # Step 1: Segment into sentences
        sentences = segment_article_into_sentences(article)
        #print("Sents: ",sentences)
        # Step 2: Classify which sentences are factual claims
        claims = classify_sentences_as_claims(sentences)
        print("Claims: ",claims)
        return jsonify({"claims": claims})

    except Exception as e:
        import traceback2 as traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500




def decompose_claim(claim: str):
    prompt = f"""
            You are a fact-checking assistant. Generate 5 yes or no questions to help me answer if the given claim is true or false.

Claim: "{claim}"

Subquestions:
1.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7,
        stop=None
    )
    questions = response.choices[0].message.content.strip().split("\n")
    return [q.strip("0123456789. ").strip() for q in questions if q.strip()]

from readability.readability import Document
import requests
from bs4 import BeautifulSoup

def clean_html_input(raw_html):
    # Remove NULL bytes and control characters
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", raw_html)
    return cleaned

def extract_readable_text(url):
    raw_html = ""
    try:
        session = requests.Session()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0 Safari/537.36"
            )
        }
        response = session.get(url, headers=headers, timeout=10)

        if response.status_code == 403:
            print(f"[403] Forbidden (likely anti-bot): {url}")
            return None
        if response.status_code != 200:
            print(f"[{response.status_code}] Cannot access: {url}")
            return None
        
        raw_html = response.text
        html = clean_html_input(raw_html)  # ✅ clean it before parsing
        doc = Document(html)
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)

        if len(text.split()) < 30:
            print(f"[Skipped] Too short: {url}")
            return None

        return text

    except Exception as e:
        print(f"[Error parsing HTML for {url}]: {e}")
        with open("bad_html_log.txt", "a") as f:
            f.write(f"URL: {url}\n{raw_html[:500]}\n\n")
        return None
    
from newspaper import Article

def extract_with_newspaper(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        if len(text.split()) < 30:
            print(f"[Skipped] Too short: {url}")
            return None
        return text
    except Exception as e:
        print(f"[Error] {url} -> {e}")
        return None
def extract_readable_text_combined(url):
    text = extract_readable_text(url)
    if text:
        return text
    return extract_with_newspaper(url)
import requests

def google_custom_search(query, serp_api, cx, num_results=5):
    if not query.strip():
        raise ValueError("Google search query is empty!")
    #search_url = "https://www.googleapis.com/customsearch/v1"
    search_url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "api_key": serp_api,
        "q": query,
        "num": num_results
    }
    response = requests.get(search_url, params=params)
    response.raise_for_status()
    results = response.json()

    links = []
    for result in results.get("organic_results", []):
        links.append({
            "title": result.get("title"),
            "link": result.get("link"),
            "snippet": result.get("snippet", "")
        })
    return links

####Use of NewsAPI here:
import requests
from datetime import datetime, timedelta

def newsapi_search(query: str,
                   newsapi_key: str,
                   from_date: str | None = None,
                   to_date: str | None = None,
                   language: str = "en",
                   page_size: int = 10,
                   sort_by: str = "relevancy"):
    """
    Wraps NewsAPI 'everything' endpoint and returns a list of {title, link, snippet, source, publishedAt}.
    Docs: https://newsapi.org/docs/endpoints/everything
    """
    if not query.strip():
        return []

    endpoint = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": language,
        "pageSize": page_size,
        "sortBy": sort_by,
        "apiKey": newsapi_key,
    }
    if from_date:
        params["from"] = from_date  # e.g., "2023-01-01"
    if to_date:
        params["to"] = to_date

    try:
        r = requests.get(endpoint, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[NewsAPI error] {e}")
        return []

    results = []
    for art in data.get("articles", []):
        results.append({
            "title": art.get("title", ""),
            "link": art.get("url", ""),
            "snippet": art.get("description", "") or "",
            "source": (art.get("source") or {}).get("name", ""),
            "publishedAt": art.get("publishedAt", "")
        })
    return results

def build_time_window(claim_date: datetime | None, days_window: int = 365):
    if not claim_date:
        return None, None
    from_date = (claim_date - timedelta(days=days_window)).strftime("%Y-%m-%d")
    to_date = claim_date.strftime("%Y-%m-%d")
    return from_date, to_date

from urllib.parse import urlparse

def _norm_url(u: str) -> str:
    try:
        p = urlparse(u)
        return f"{p.scheme}://{p.netloc}{p.path}"
    except Exception:
        return u

def merge_and_dedup_results(*lists):
    seen = set()
    merged = []
    for L in lists:
        for item in L:
            link = _norm_url(item.get("link", ""))
            if not link or link in seen:
                continue
            seen.add(link)
            merged.append(item)
    return merged

# optional: prefer trusted news domains up front
TRUSTED_DOMAINS = {
    "reuters.com": 2.0, "apnews.com": 2.0, "bbc.com": 1.8, "nytimes.com": 1.6,
    "theguardian.com": 1.6, "ft.com": 1.6, "wsj.com": 1.6, "washingtonpost.com": 1.5
}

def domain_weight(url: str) -> float:
    try:
        host = urlparse(url).netloc or ""
        host = host.lower()
        for dom, w in TRUSTED_DOMAINS.items():
            if host.endswith(dom):
                return w
    except:
        pass
    return 1.0



from rank_bm25 import BM25Okapi
def bm25_rerank(query, documents, k1=30, k2=150, top_k=4):
    chunks = []
    for doc in documents:
        words = nltk.word_tokenize(doc)
        for i in range(0, len(words), k1 // 2):
            chunk = words[i:i+k1]
            chunks.append(" ".join(chunk))

    tokenized_chunks = [nltk.word_tokenize(c.lower()) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(nltk.word_tokenize(query.lower()))
    top_chunks = [chunks[i] for i in sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]]
    return top_chunks
def summarize_claim_focused(claim: str, evidence: str):
    prompt = f"""
Suppose you are assisting a fact-checker to fact-check the claim.

Claim: "{claim}"

Document:
\"\"\"
{evidence}
\"\"\"

Summarize the relevant information from the document in 1-2
sentences. Your response should provide a clear and concise
summary of the relevant information contained in the document.
Do not include a judgment about the claim and do not repeat any
information from the claim that is not supported by the
document.
Summarization:

"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()
def gpt4_classify_veracity(claim, evidence_summary):
    prompt = f"""
You are a professional fact-checking assistant. Your task is to read a factual claim and a summary of supporting or opposing evidence, and classify the claim's truthfulness into one of the following 6 categories:

- true
- mostly true
- half true
- barely true
- false
- pants-on-fire

### Guidelines:
- "true": The evidence clearly confirms all factual aspects of the claim.
- "mostly true": The evidence confirms most aspects, with minor issues or missing context.
- "half true": The evidence is mixed, with significant confirmations and contradictions.
- "barely true": Only a small part of the claim is supported by evidence.
- "false": The claim is directly contradicted by the evidence.
- "pants-on-fire": The claim is not only false but wildly inaccurate or fabricated.

### Examples:

**Claim**: The U.S. has the highest number of gun deaths in the world.
**Evidence**: The U.S. ranks 32nd in gun deaths per capita globally. Countries like El Salvador, Venezuela, and Honduras have much higher rates. However, the U.S. does lead in total number of gun deaths due to population size.
**Classification**: half true

**Claim**: The moon landing was faked and filmed in a studio.
**Evidence**: There is overwhelming scientific, photographic, and third-party evidence confirming the Apollo 11 landing in 1969. The conspiracy theory lacks credible support.
**Classification**: pants-on-fire

---

Now evaluate this case:

**Claim**: {claim}

**Evidence**:
{evidence_summary}

**Classification**:"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=20
    )

    label = response.choices[0].message.content.strip().lower()
    return label

### Single claim verification route
@app.route("/verify-claim", methods=["POST"])
def verify_claim():
    try:
        data = request.json
        claim = data.get("claim", "")
        if not claim.strip():
            return jsonify({"error": "Missing or empty claim"}), 400

        # Step 1: Decompose claim into subquestions
        subqs = decompose_claim(claim)
        subqs = [q for q in subqs if q.strip()]

        #claim_date = None  # datetime(...) if you have it
        from_date, to_date = (None, None)

        # Step 2: Retrieve documents from web
        documents = []
        raw_results_all = []
        for subq in subqs:
            if not subq.strip():
                continue
            # A) general web (SerpAPI)
            serp_results = google_custom_search(subq, serp_api, cx) #serp results
            # B) news-only (NewsAPI)
            news_results = newsapi_search(
                query=subq,
                newsapi_key=NEWSAPI_KEY,
                from_date=from_date,
                to_date=to_date,
                language=DEFAULT_LANGUAGE,
                page_size=10,
                sort_by="relevancy",
            )
            # Merge + dedup
            merged = merge_and_dedup_results(serp_results, news_results)
            # Prefer trusted domains (simple front-weighting)
            merged.sort(key=lambda r: domain_weight(r.get("link","")), reverse=True)
            # Collect snippets or scrape text
            for result in merged:
                #url = result["link"]
                url = result.get("link", "")
                #snippet = result.get("snippet", "")
                snippet = (result.get("snippet") or "").strip()
                
                # First try snippet, fallback to scraping
                if snippet and len(snippet.split()) > 10:
                    documents.append(snippet)
                else:
                    text = extract_readable_text_combined(url)
                    if text:
                        documents.append(text)
            raw_results_all.extend(merged)

        if not documents:
            return jsonify({
                "claim": claim,
                "label": "insufficient evidence",
                "summary": "",
                "note": "No evidence documents could be retrieved."
            })

        # Step 3: BM25 Reranking
        top_chunks = []
        for subq in subqs:
            top_chunks.extend(bm25_rerank(subq, documents))
        evidence_context = "\n\n".join(top_chunks)

        # Step 4: Summarize
        final_summary = summarize_claim_focused(claim, evidence_context)

        # Step 5: Classify
        label = gpt4_classify_veracity(claim, final_summary)

        evidence_preview = [{
            "title": r.get("title",""),
            "link": r.get("link",""),
            "source": r.get("source",""),
            "publishedAt": r.get("publishedAt","")
        } for r in raw_results_all[:6]]

        return jsonify({
            "claim": claim,
            "label": label,
            "summary": final_summary,
            "evidence_sources": evidence_preview
        })

    except Exception as e:
        import traceback2 as traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    

#------Route for verified Claim Re-writing---------------
def find_best_sentence(article, claim, cutoff = 65):
        sentences = sent_tokenize(article)
        # pick the article sentence closest to the claim
        match = process.extractOne(
            claim,
            sentences,
            scorer=fuzz.token_set_ratio
        )
        if match and match[1] >= cutoff:
            best_sentence = match[0]
            idx = sentences.index(best_sentence)
            return best_sentence, idx
        return None, None
@app.route("/propose-correction", methods=["POST"])
def propose_correction():
    try:
        data = request.json
        claim = data.get("claim", "").strip()
        label = data.get("label", "").strip().lower()
        summary = data.get("summary", "").strip()
        article = data.get("article", "").strip()

        if not claim or not article:
            return jsonify({"error": "Missing claim or article"}), 400

        # Only propose corrections when the verdict suggests it's wrong or mostly wrong
        if label not in {"false", "pants-on-fire", "barely true"}:
            return jsonify({
                "can_edit": False,
                "reason": "Label not strongly negative; skipping correction proposal."
            })

        target_sentence, sent_idx = find_best_sentence(article, claim)
        if not target_sentence:
            return jsonify({
                "can_edit": False,
                "reason": "Could not locate a matching sentence in the article."
            })

        # Ask GPT-4 to write a corrected sentence grounded in evidence summary
        prompt = f"""
                    You revise news text for factual correctness. Based on the evidence summary, rewrite the target sentence so it is accurate, neutral, and concise.
                    - Keep the *same topic and role in the paragraph*.
                    - Do not add new claims beyond the evidence summary.
                    - Prefer attribution if needed (e.g., "Analysts say..." or "According to X...").

                    Claim (incorrect): {claim}

                    Evidence summary:
                    {summary}

                    Original sentence (from the article to replace):
                    "{target_sentence}"

                    Return ONLY the corrected sentence (one sentence).
                    """
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=120
        )
        corrected_sentence = response.choices[0].message.content.strip().strip('"')

        return jsonify({
            "can_edit": True,
            "target_sentence": target_sentence,
            "corrected_sentence": corrected_sentence,
            "sentence_index": sent_idx
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route("/apply-edit", methods=["POST"])
def apply_edit():
    try:
        data = request.json
        article = data.get("article", "")
        target_sentence = data.get("target_sentence", "")
        corrected_sentence = data.get("corrected_sentence", "")

        if not (article and target_sentence and corrected_sentence):
            return jsonify({"error": "Missing article/target_sentence/corrected_sentence"}), 400

        if target_sentence not in article:
            # fallback: try fuzzy replace on the best sentence
            best, _ = find_best_sentence(article, target_sentence, cutoff=60)
            if not best:
                return jsonify({"error": "Could not locate the sentence to replace."}), 400
            target_sentence = best

        updated_article = article.replace(target_sentence, corrected_sentence, 1)

        return jsonify({
            "updated_article": updated_article,
            "replaced": True
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500




# --- Route for overall claim  verification---
# @app.route("/fact-verify", methods=["POST"])
# def fact_verify():
#     try:
#         data = request.json
#         article = data["article"]
#         if not article:
#             return jsonify({"error": "Missing article in request"}), 400
        
#         # Step 1: Segment and classify claims
#         sentences = segment_article_into_sentences(article)
#         claims = classify_sentences_as_claims(sentences)

#         results = []
#         for claim in claims:
#             # Step 2: Decompose the claim into subquestions
#             subqs = decompose_claim(claim)
#             # Filter out empty or whitespace-only subquestions
#             subqs = [q for q in subqs if q.strip()]

#             # Step 3: Retrieve documents
#             documents = []
#             for subq in subqs:
#                 if not subq.strip():
#                     continue  # skip invalid queries
#                 urls = google_custom_search(subq, google_api, cx)
#                 for result in urls:
#                     url = result["link"]
#                     text = extract_readable_text_combined(url)
#                 if text and len(text.split()) >= 30:
#                     documents.append(text)
#                 else:
#                     snippet = result.get("snippet", "").strip()
#                     if snippet:
#                         print(f"[Fallback to snippet] {url}")
#                         documents.append(snippet)

#             if not documents:
#                 results.append({
#                     "claim": claim,
#                     "label": "insufficient evidence",
#                     "summary": "",
#                     "note": "No documents could be retrieved."
#                 })
#                 continue

#             # Step 4: BM25 rerank and build context
#             top_chunks = []
#             for subq in subqs:
#                 top_chunks.extend(bm25_rerank(subq, documents))
#             evidence_context = "\n\n".join(top_chunks)

#             # Step 5: Summarize evidence
#             final_summary = summarize_claim_focused(claim, evidence_context)

#             # Step 6: Classify claim
#             label = gpt4_classify_veracity(claim, final_summary)

#             results.append({
#                 "claim": claim,
#                 "label": label,
#                 "summary": final_summary
#             })

#         return jsonify({"verifications": results})
#     except Exception as e:
#         import traceback2 as traceback
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500





""" @api.route('/topics/Mistral/summaries', methods=['GET'])
class Summaries(Resource):
    def get(self):
        try:
            # Fetch summaries asynchronously
            async def fetch_cached_summaries():
                summaries = await redis_client.get("mistral_summaries")
                print("summaries Ok")
                #if summaries:
                    #await redis_client.delete("mistral_summaries")  # Delete the key after retrieval
                return json.loads(summaries) if summaries else None

            cached_summaries = asyncio.run(fetch_cached_summaries())
            print("cached summaries:")
            print(cached_summaries)

            if not cached_summaries:
                print("Summaries not found in Redis.")
                return {"status": "Summaries are still being processed. Please try again later."}, 202

            print("Retrieved summaries from Redis:", cached_summaries)  # Debug log
            return jsonify({"summaries": cached_summaries})
        except Exception as e:
            print(f"Error retrieving summaries: {e}")
            return {"error": f"Failed to retrieve summaries: {e}"}, 400 """

""" @api.route('/generate-news', methods=['POST'])
class GenerateNews(Resource):
    def post(self):
        try:
            # Parse the request body
            data = request.json
            topic = data.get("topic")
            summary = data.get("summary")
            keywords = data.get("keywordsSent")
            style = data.get("style")

            # Fetch merged summary from Redis
            async def fetch_merged():
                key = f"merged_summary:{topic}"
                result = await redis_client.get(key)
                return json.loads(result) if result else None

            summary = asyncio.run(fetch_merged())
            if not summary:
                return {"error": "No merged summary found for this topic"}, 404
            

            # Generate the news article
            article = mistral_modeling.generate_news_article(topic, summary, keywords, style)
            print("Generated news:", article)

            return jsonify({"article": article})
        except Exception as e:
            return {"error": str(e)}, 400 """
        




""" @api.route('/summarize')
class Summarize(Resource):
    @api.doc(body=api.model(
        'Summarize', {
            'conversations': fields.List(fields.String, description='list of conversations')
        }
    ))
    def post(self):
        conversation_list = request.json["conversations"]
        conv_summaries = summarizer_agent.run_all(conversation_list)
        return jsonify({"summaries": conv_summaries}) """


""" @api.route('/topic-aware-summarize')
class TopicAwareSummarize(Resource):
    @api.doc(body=api.model(
        'TopicAwareModel', {
            'conversations': fields.List(fields.String, description='list of conversations'),
            'dialogue': fields.ClassName('Samsum', description='dialogue to summarize'),
            'num_topics': fields.Integer(description='number of topics to extract')
        }
    ))
    def post(self):
        conversation_list = request.json["conversations"]
        dialogue_to_summarize = request.json["dialogue"]
        num_topics = int(request.json["num_topics"])
        user_topics = request.json.get("user_topics", [])
        # Update topic count if necessary.
        bertopic.check_topic_count(num_topics)
        
        try:
            if len(user_topics) > 0:
                topics_df, topic_embeddings = bertopic.get_input_topic_embeddings(
                    user_topics)
            else:
                topics_df, topic_embeddings = bertopic.get_topic_embeddings(
                    conversation_list)
                if len(topic_embeddings < 3):
                    topics_df, topic_embeddings = bertopic.get_static_topics()
        except:
            topics_df, topic_embeddings = bertopic.get_static_topics()
        dict_topic_sentences = topic_aware_summarizer.extract_topic_sentences(dialogue_to_summarize, topics_df,
                                                                              topic_embeddings)
        conv_summaries = summarizer_agent.run_all_topic_aware(
            dict_topic_sentences)
        return jsonify({"conv_summaries": conv_summaries}) """

""" @api.route('/topic-aware-summarize')
class TopicAwareSummarize(Resource):
    @api.doc(body=api.model(
        'TopicAwareModel', {
            'conversations': fields.List(fields.String, description='list of conversations')
        }
    ))
    def post(self):
        try:
            conversation_list = request.json["conversations"]
            combined_conversations = " ".join(conversation_list)

            # Retrieve cached topics and keywords
            async def fetch_cached_topics():
                cached_data = await redis_client.get("mistral_topics")
                return json.loads(cached_data) if cached_data else None

            cached_topics = asyncio.run(fetch_cached_topics())
            if not cached_topics:
                return {"error": "No cached topics found. Please perform topic modeling first."}, 400

            # Perform summarization
            topics_and_keywords = cached_topics["topics_and_keywords"]
            # summaries = mistral_modeling.summarize_with_hint_test(combined_conversations, topics_and_keywords)
            summaries = mistral_modeling.summarize_with_hint(combined_conversations, topics_and_keywords)

            return jsonify({"summaries": summaries})
        except Exception as e:
            return {"error": str(e)}, 400
 """


""" @api.route('/delta-summarize')
class DeltaSummarize(Resource):
    @api.doc(body=api.model(
        'DeltaSummarize', {
            'summaries': fields.List(fields.String, description='list of summaries'),
            'plot_type': fields.String(description='type of plot'),
            'dialogue': fields.ClassName('Samsum', description='dialogue to summarize'),
            'default_summary': fields.String(description='default summary')
        }
    ))
    def post(self):
        summaries = request.json["summaries"]
        plot_type = request.json["plot_type"]
        dialogue = request.json["dialogue"]
        default_summary = request.json["default_summary"]
        plot_img = delta_summarizer.send_plot(
            plot_type, summaries, dialogue, default_summary)
        return send_file(plot_img, mimetype='image/png') """


@app.route('/health')
class Health(Resource):
    @api.doc(description='Check if the app is running.')
    def get(self):
        """
        API endpoint to check if the app has started running
        :return: The health status of the app.
        """
        return jsonify({"status": "healthy"})


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers',
                         'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


# Error handlers
@app.errorhandler(404)
def handle_not_found(error):
    return jsonify({"message": error.description}), 404


@app.errorhandler(Exception)
def handle_server_error(error):
    return jsonify({"message": "Internal server error: {}".format(error)}), 500
