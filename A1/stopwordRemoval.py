from A1.util import *

# Add your import statements here
from nltk.corpus import stopwords

nltk.download('stopwords')

class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = []
		stop_words = set(stopwords.words('english'))
		for sentence in text:
			fitered_sent = [word for word in sentence if word.lower() not in stop_words]
			stopwordRemovedText.append(fitered_sent)

		return stopwordRemovedText
	
	def fromdata(self,text,threshold = 0.5):
		num_docs = len(text)
		doc_freq = {}
		for docs in text:
			words =	 set(word for sent in docs for word in sent)
			for word in words:
				if word in doc_freq:
					doc_freq[word] += 1
				else:
					doc_freq[word] = 1

		stopwords = [word for word, freq in doc_freq.items() if freq/num_docs >= threshold]

		return stopwords
		
		





	