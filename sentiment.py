from stats_text_analysis import *
import numpy as np
import os
import json
import torch
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":

    # Set up environment
    save_path = "./result"
    data_path = "./data"
    seed = 20010410
    length = 1280 * 4
    contact = <contact_name>
    sender1 = <sender1_name>
    sender2 = <sender2_name>
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # Tasks: "chats-correlation", "llm-vs-bayesian", "chats-correlation-by-time"
    task = "chats-correlation"
    print("Loading chat data...")
    with open(os.path.join(data_path, "temp.json"), "r", encoding="utf-8") as f:
        chats_json = json.load(f)
    chats_data = chats_json.get("chats").get("list")
    # Load chat data
    if task == "llm-vs-bayesian":
        chats_dataloader = load_data(chats_data, contact, sender1, length=length)
        sentiment_llm_all = pd.DataFrame(sentiment_analysis_llm(chats_dataloader, save_path=save_path))
        sentiment_llm = np.asarray(sentiment_llm_all["positive"])

        sentiment_traditional = sentiment_analysis_bayesian(chats_dataloader, save_path, lang="zh", train=False)
        sentiment_traditional = np.asarray(sentiment_traditional)

        # Perform statistical analysis of sentiment scores by LLM and traditional methods
        perform_statistical_analysis(sentiment_llm, sentiment_traditional)
        
    elif task == "chats-correlation":

        two_chats_dataloader = load_data(chats_data, contact, "both", length=length)

        sentiment_llm1, sentiment_llm2 = sentiment_analysis_llm(
            two_chats_dataloader, mode="both", save_path=save_path
        )
        sentiment_llm1 = np.asarray(sentiment_llm1["positive"])
        sentiment_llm2 = np.asarray(sentiment_llm2["positive"])
        
        perform_statistical_analysis(sentiment_llm1, sentiment_llm2)
    elif task == "chats-correlation-by-time":
        two_chats_dataloader = load_data(chats_data, contact, "both", length=50)
        sentiment_llm1, sentiment_llm2 = sentiment_analysis_llm(
            two_chats_dataloader, mode="both-by-time", save_path=save_path
        )
