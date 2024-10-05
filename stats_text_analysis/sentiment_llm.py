import os
import torch
from transformers import pipeline
from tqdm import tqdm
from .utils import plot_sentiment_distribution_all, plot_all_sentiments
import matplotlib.pyplot as plt


def sentiment_analysis_llm(chats_dataloader, save_path, mode="single"):
    if mode == "single":
        # Load sentiment analysis model
        print("Loading sentiment analysis model...")
        distilled_student_sentiment_classifier = pipeline(
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1,
        )

        # Perform sentiment analysis by LLM
        print("Performing sentiment analysis by LLM...")
        sentiment_llm = []
        for idx, chat in tqdm(enumerate(chats_dataloader)):
            sentiment = distilled_student_sentiment_classifier(chat.get("messages"))
            sentiment_llm += sentiment
        sentiment_scores_llm = {
            "positive": [],
            "neutral": [],
            "negative": [],
        }
        for idx, sentiment in enumerate(sentiment_llm):
            sentiment_scores_llm["positive"].append(sentiment[0].get("score"))
            sentiment_scores_llm["neutral"].append(sentiment[1].get("score"))
            sentiment_scores_llm["negative"].append(sentiment[2].get("score"))
        # print(sentiment_scores_llm)

        plot_sentiment_distribution_all(sentiment_scores_llm, "LLM", save_path)
        plot_all_sentiments(sentiment_scores_llm, "LLM", save_path)

        return sentiment_scores_llm

    elif "both" in mode:
        # Load sentiment analysis model
        print("Loading sentiment analysis model...")
        distilled_student_sentiment_classifier = pipeline(
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1,
        )

        # Perform sentiment analysis by LLM
        print("Performing sentiment analysis by LLM...")
        sentiment_llm_sender1 = []
        sentiment_llm_sender2 = []
        sender1 = ""
        sender2 = ""
        sender1_time = []
        sender2_time = []
        for idx, chat in tqdm(enumerate(chats_dataloader)):
            sentiment = distilled_student_sentiment_classifier(chat.get("messages"))
            for i, chat_sender in enumerate(chat.get("sender")):
                if not sender1:
                    sender1 = chat_sender
                if chat_sender == sender1:
                    sentiment_llm_sender1 += [sentiment[i]]
                    sender1_time += [chat.get("dates")[i]]
                else:
                    if not sender2:
                        sender2 = chat_sender
                    sentiment_llm_sender2 += [sentiment[i]]
                    sender2_time += [chat.get("dates")[i]]
        sentiment_scores_llm_sender1 = {
            "sender": sender1,
            "positive": [],
            "neutral": [],
            "negative": [],
            "time": sender1_time,
        }
        sentiment_scores_llm_sender2 = {
            "sender": sender2,
            "positive": [],
            "neutral": [],
            "negative": [],
            "time": sender2_time,
        }
        
        for idx, sentiment in enumerate(sentiment_llm_sender1):
            sentiment_scores_llm_sender1["positive"].append(sentiment[0].get("score"))
            sentiment_scores_llm_sender1["neutral"].append(sentiment[1].get("score"))
            sentiment_scores_llm_sender1["negative"].append(sentiment[2].get("score"))
        for idx, sentiment in enumerate(sentiment_llm_sender2):
            sentiment_scores_llm_sender2["positive"].append(sentiment[0].get("score"))
            sentiment_scores_llm_sender2["neutral"].append(sentiment[1].get("score"))
            sentiment_scores_llm_sender2["negative"].append(sentiment[2].get("score"))
        # print(sentiment_scores_llm)

        if mode == "both":
            plot_sentiment_distribution_all(
                sentiment_scores_llm_sender1, f"LLM {sender1}", save_path
            )
            plot_all_sentiments(
                sentiment_scores_llm_sender1, f"LLM {sender1}", save_path
            )
            plot_sentiment_distribution_all(
                sentiment_scores_llm_sender2, f"LLM {sender2}", save_path
            )
            plot_all_sentiments(
                sentiment_scores_llm_sender2, f"LLM {sender2}", save_path
            )
        elif mode == "both-by-time":
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(sender1_time)), sentiment_scores_llm_sender1["positive"], label=sentiment_scores_llm_sender1["sender"])
            plt.plot(range(len(sender2_time)), sentiment_scores_llm_sender2["positive"], label=sentiment_scores_llm_sender2["sender"])
            print(sentiment_scores_llm_sender1["sender"], sentiment_scores_llm_sender2["sender"])
            plt.legend()
            plt.savefig(os.path.join(save_path, f"sentiment_change_positive_llm.png"))
        return (sentiment_scores_llm_sender1, sentiment_scores_llm_sender2)