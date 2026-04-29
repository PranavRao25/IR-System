from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, WordNetLemmatizer, PorterStemmer, etc.)
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

class InflectionReduction:

	def porterStemmer(self, text):
		"""
		Inflection Reduction using Porter Stemmer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed tokens representing a sentence
		"""

		reducedText = []

		# Fill in code here
		stemmer = PorterStemmer()
		for sentence in text:
			stemmed_sent = [stemmer.stem(word) for word in sentence]
			reducedText.append(stemmed_sent)

		return reducedText



	def wordnetLemmatizer(self, text):
		"""
		Inflection Reduction using WordNet Lemmatizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			lemmatized tokens representing a sentence
		"""

		reducedText = [	]

		# Fill in code here
		lemmatizer = WordNetLemmatizer()
		for sentence in text:
			lemmatized_sent = [lemmatizer.lemmatize(word) for word in sentence]
			reducedText.append(lemmatized_sent)

		return reducedText



	def reduce(self, text):
		"""
		Wrapper function for inflection reduction.
		Students may choose which method to call
		or extend this function to support both options.
		"""

		reducedText = self.wordnetLemmatizer(text)
		return reducedText
