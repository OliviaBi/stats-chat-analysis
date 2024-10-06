# stats-chat-analysis
Analyze and visualize the trend and distribution of the sentiment of chats 


## Setup

Install the required packages by running

`pip install -r requirements.txt`

## Running

Run analysis and visualization with the following steps:
 - Export your telegram chats in the JSON format and save it under `data` folder
 - Set `<contact_name>, <sender1_name>, <sender2_name>` and other parameters you want to customize in `sentiment.py`
 - Set the task you want to implement
 - Run `python sentiment.py` and obtain the results from `result` folder

## Tasks

### llm-vs-bayesian

Goal: Examnine whether distributions of sentiment extracted from methods based on deep learning and traditional Bayesian inference have significant difference

  - Analyze the sentiment of chats from one sender by both Multilingual DistilBERT and Bayesion inference
  - Visualize the distributions of sentiment from both methods and fit a continuous probability density curve with kernel density estimation
  - Conduct Kolmogorov-Smirnov test on both distributions

### chats-correlation

Goal: Examnine whether distributions of sentiment of two chatters have significant difference or similarity

  - Analyze the sentiment of two people in the same chat with Multilingual DistilBERT
  - Visualize the distributions of sentiment from both people and fit a continuous probability density curve with kernel density estimation
  - Conduct Kolmogorov-Smirnov test on both distributions

### chats-correlation-by-time

Goal: Examnine the correlation and trends of two chatters to evaluate whether the sentiment of them, influenced by each other during chatting, will have a similar sentiment change trend

  - Analyze the sentiment of two people in the same chat with Multilingual DistilBERT
  - Visualize the distributions of sentiment from both people and fit a continuous probability density curve with kernel density estimation
  - Conduct Kolmogorov-Smirnov test on both distributions

## Example Results

Based on personal chat history, the result of llm-vs-bayesion rejects the hypothesis that sentiment extracted by deep learning and Bayesian inference follows the same distribution when $\alpha$ is set as 0.05:

### Distribution of Sentiment

![from llm](https://github.com/OliviaBi/stats-chat-analysis/blob/main/resources/sentiment_distribution_Positive_LLM.png)
![from bayesian](https://github.com/OliviaBi/stats-chat-analysis/blob/main/resources/sentiment_distribution_Traditional.png)

### K-S Test Result with $\alpha=0.05$

![KS test](https://github.com/OliviaBi/stats-chat-analysis/blob/main/resources/demo-result.png)

### Possible Reasons:

  - Differences between the training data of DistilBERT (from [Amazon customer reviews](https://huggingface.co/datasets/tyqiangz/multilingual-sentiments)) and Bayesian inference (from book reviews);
  - Performance gaps of models' analysis ability;
  - The selection of Bayesian inference's priori hypothesis can have great influence on its performance;
  - The independence hypothesis of Naive Bayes may not awalys hold true, especially in chats that have rich context;
  - The generalization ability of DistilBERT is better than Bayesian inference since Bayesian inference relies heavily on the priori and feature representation, especially under complex and everchanging text condition, like numerous daily chats containing various topics;
  - The Vapnik-Chervonenkis dimension of Bayesian method is low due to its simplicity, making its generalization ability heavily rely on the consistency between the data and priori.

## More Work In Progress

Integrate the code to a web application. To be released soon.
