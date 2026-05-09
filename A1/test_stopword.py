import json
from stopwordRemoval import StopwordRemoval
import nltk
from nltk.corpus import stopwords

stp_rm = StopwordRemoval()

with open("output/tokenized_docs.txt", 'r') as f:
    tokenized_docs = json.load(f)


nltk_stopwords = set(stopwords.words('english'))
data_stopwords = stp_rm.fromdata(tokenized_docs)

overlap = len(set(data_stopwords).intersection(nltk_stopwords))
num_stopwords = len(data_stopwords)
words_in_data = len(set(word for word in data_stopwords if word not in nltk_stopwords))

print(f"Number of stopwords identified: {num_stopwords}")
print(f"Overlap with NLTK stopwords: {overlap}")
print(f"Words that appear in one list but not the other: {words_in_data}")


