import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from snownlp import SnowNLP
from tqdm import tqdm
from stats_text_analysis.utils import plot_sentiment_distribution_all, plot_all_sentiments, plot_sentiment_distribution



def sentiment_analysis_bayesian(chats_dataloader, save_path, lang="en", train=False):
    # Perform sentiment analysis by Traditional Methods
    print("Performing sentiment analysis by Bayesian Inference...")
    if lang == "en":
        nltk.download("vader_lexicon")
        analyzer = SentimentIntensityAnalyzer()
        sentiment_nltk = []

        for idx, chat in tqdm(enumerate(chats_dataloader)):
            sentiment = []
            for message in chat.get("messages"):
                score = analyzer.polarity_scores(message)
                print(score)
                print(message)
                exit()
                sentiment.append(score)
            sentiment_nltk += sentiment

        sentiment_scores_nltk = {
            "positive": [],
            "neutral": [],
            "negative": [],
        }

        for sentiment in sentiment_nltk:
            sentiment_scores_nltk["positive"].append(sentiment.get("pos"))
            sentiment_scores_nltk["neutral"].append(sentiment.get("neu"))
            sentiment_scores_nltk["negative"].append(sentiment.get("neg"))

        print(sentiment_scores_nltk)

        plot_sentiment_distribution_all(sentiment_scores_nltk, "Traditional", save_path)
        plot_all_sentiments(sentiment_scores_nltk, "Traditional", save_path)
        return sentiment_scores_nltk

    elif lang == "zh":
        sentiment_snownlp = []
        for idx, chat in tqdm(enumerate(chats_dataloader)):
            for message in chat.get("messages"):
                score = SnowNLP(message).sentiments
                sentiment_snownlp.append(score)
        plot_sentiment_distribution(
            sentiment_snownlp, "green", "Traditional", save_path, bins=20
        )
        return sentiment_snownlp
