import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_chats_by_name(chat_list, name):
    for chat in chat_list:
        if chat.get("name") == name:
            return chat
    return None


def get_chats_by_sender(chat_messages, sender, start="rand", limit=1280 * 4):
    if sender != "both":
        chats = {"date": [], "text": []}
        if start == "rand" and limit != -1:
            start = np.random.randint(0, len(chat_messages) - limit)
        elif limit == -1:
            limit = len(chat_messages)
            start = 0
        print(f"Start: {start}, Limit: {limit}")
        for idx, chat in tqdm(enumerate(chat_messages[start : start + limit])):
            if (
                chat.get("from")
                and not chat.get("forwarded_from")
                and chat.get("from").split(" ")[0] == sender
                and chat.get("text")
                and type(chat.get("text")) == str
            ):
                chats["date"].append(chat.get("date"))
                chats["text"].append(chat.get("text"))
        return chats
    else:
        chats = {"date": [], "text": [], "sender": []}
        if start == "rand" and limit != -1:
            start = np.random.randint(0, len(chat_messages) - limit)
        elif limit == -1:
            limit = len(chat_messages)
            start = 0
        print(f"Start: {start}, Limit: {limit}")
        for idx, chat in tqdm(enumerate(chat_messages[start : start + limit])):
            if (
                chat.get("from")
                and not chat.get("forwarded_from")
                and chat.get("text")
                and type(chat.get("text")) == str
            ):
                chats["date"].append(chat.get("date"))
                chats["text"].append(chat.get("text"))
                chats["sender"].append(chat.get("from").split(" ")[0])
        return chats


def plot_sentiment_distribution(
    sentiment_scores, color, method, path, sentiment=None, bins=20
):
    if sentiment:
        sns.displot(
            sentiment_scores,
            kde=True,
            color=color,
            label=sentiment,
            stat="density",
            bins=bins,
        )
        plt.title(f"Sentiment Distribution ({sentiment}) by {method}")
        plt.savefig(
            os.path.join(path, f"sentiment_distribution_{sentiment}_{method}.png"),
            bbox_inches="tight",
        )
    else:
        sns.displot(
            sentiment_scores,
            kde=True,
            color=color,
            label="Sentiment",
            stat="density",
            bins=bins,
        )
        plt.title(f"Sentiment Distribution by {method}")
        plt.savefig(
            os.path.join(path, f"sentiment_distribution_{method}.png"),
            bbox_inches="tight",
        )


def plot_sentiment_distribution_all(sentiment_scores, method, path, bins=20):
    plot_sentiment_distribution(
        sentiment_scores["positive"],
        "green",
        method,
        path,
        sentiment="Positive",
        bins=bins,
    )
    plot_sentiment_distribution(
        sentiment_scores["neutral"],
        "blue",
        method,
        path,
        sentiment="Neutral",
        bins=bins,
    )
    plot_sentiment_distribution(
        sentiment_scores["negative"],
        "red",
        method,
        path,
        sentiment="Negative",
        bins=bins,
    )


def plot_all_sentiments(sentiment_scores, method, path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        sentiment_scores["positive"],
        sentiment_scores["neutral"],
        sentiment_scores["negative"],
        c=np.asarray(sentiment_scores["positive"]),
        cmap="viridis",
        marker="o",
    )
    ax.set_xlabel("Positive")
    ax.set_ylabel("Neutral")
    ax.set_zlabel("Negative")
    plt.title(f"3D Scatter Plot of Sentiment Scores by {method}")
    plt.savefig(os.path.join(path, f"sentiment_distribution_3d_{method}.png"))

