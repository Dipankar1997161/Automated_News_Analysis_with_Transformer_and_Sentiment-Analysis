#!/usr/bin/env python
# coding: utf-8

# ## Libraries and setup

get_ipython().system('pip install sentencepiece')

from bs4 import BeautifulSoup
import requests
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained("human-centered-summarization/financial-summarization-pegasus")
model = PegasusForConditionalGeneration.from_pretrained(model_name)

model.config.vocab_size ## Check transformer model vocab size

### Scrap the web

URL = "https://au.finance.yahoo.com/news/abl-space-systems-scores-60-204923372.html"
r = requests.get(URL)
soup = BeautifulSoup(r.text, 'html.parser')
paragraphs = soup.find_all('p')

### Clean the paragraph and Quick summary

text = [paragraph.text for paragraph in paragraphs]
words = ' '.join(text).split(' ')[:400]
article = ' '.join(words)

input_ids = tokenizer.encode(article, return_tensors = 'pt')
output = model.generate(input_ids, max_length = 70, num_beams = 3, early_stopping = True) ## Beam Search
summary = tokenizer.decode(output[0], skip_special_tokens = True)
print(summary)
# ## Automated latest NEWS Search and Summary

search_tickers = ['GME', 'TSLA', 'BTC']

def search_from_tickers(ticker):
    search_url = "https://www.google.com/search?q=yahoo+finance+()&tbm=nws".format(ticker)
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, 'html.parser')
    atags = soup.find_all('a')
    hrefs = [link['href'] for link in atags]
    return hrefs

search_from_tickers('GME')

raw_url = {ticker:search_from_tickers(ticker) for ticker in search_tickers}
raw_url['GME']

# ## Clean URLS

import re
exclude_list = ['facebook', 'subscription', 'techcrunch', 'login', 'twitter']

def strip_url(urls, exlude_list):
    val = []
    for url in urls:
        if 'https://' in url and not any(exclude_word in url for exclude_word in exclude_list):
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
    return list(set(val))


#res = re.findall(r'(https?://\S+)', url)[0].split('&')[0] - regular expression to scrap

strip_url(raw_url['GME'], exclude_list)

clean_url = {ticker:strip_url(raw_url[ticker], exclude_list) for ticker in search_tickers}
clean_url

## now we process for all the links we got like we did for the first one.

def scrap_process(URLS):
    ARTICLES = []
    for url in URLS:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = [paragraph.text for paragraph in paragraphs]
        words = ' '.join(text).split(' ')[:300]
        article = ' '.join(words)
        ARTICLES.append(article)
    return ARTICLES

articles = {ticker:scrap_process(clean_url[ticker]) for ticker in search_tickers}
print(articles)

articles["TSLA"][0]

def summarize_text(articles):
    summaries = []
    for article in articles:
        input_ids = tokenizer.encode(article, return_tensors = 'pt')
        output = model.generate(input_ids, max_length = 55, num_beams = 5, early_stopping = True)
        summary = tokenizer.decode(output[0], skip_special_tokens = True)
        summaries.append(summary)
    return summaries


summaries = {ticker:summarize_text(articles[ticker]) for ticker in search_tickers}
print(summaries)


from transformers import pipeline
sentiment = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

scores = {ticker:sentiment(summaries[ticker]) for ticker in search_tickers}
print(scores)

print(summaries['TSLA'][0], scores['TSLA'][0]['label'], scores['TSLA'][0]['score'])

def store_summary(summaries, score, urls):
    output = []
    for ticker in search_tickers:
        for count in range(len(summaries[ticker])):
            try_output = [
                ticker,
                summaries[ticker][count],
                score[ticker][count]['label'],
                score[ticker][count]['score'],
                urls[ticker][count]
            ]
            output.append(try_output)
    return output

final_output = store_summary(summaries, scores, clean_url)

print(len(final_output))
print(final_output[20: 25])

final_output.insert(0, ["Ticker", "Summary", "Label", "Confidence", "URLs"])
print(final_output)

# ## Create CSV

import csv

with open('summary_sentiment.csv', mode = 'w', newline ='') as f:
    csv_writer = csv.writer(f, delimiter =',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)

# ## Checking the file

import pandas as pd

df = pd.read_csv('summary_sentiment.csv')
df.head(10)


