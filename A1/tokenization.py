from A1.util import *

# Add your import statements here
# (Students may import required libraries such as nltk, spacy, re, etc.)
import spacy
import nltk
from nltk.tokenize import TreebankWordTokenizer

class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		# Fill in code here

		for sentence in text:
			tokens = sentence.split(" ")
			tokenizedText.append(tokens)	

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		# Fill in code here
		tokenizer = TreebankWordTokenizer()

		for sentence in text:
			tokens = tokenizer.tokenize(sentence)
			tokenizedText.append(tokens)

		return tokenizedText



	def spacyTokenizer(self, text):
		"""
		Tokenization using spaCy

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		# Fill in code here
		nlp = spacy.load("en_core_web_sm")
		for sentence in text:
			docs = nlp(sentence)
			tokens = [token.text for token in docs]
			tokenizedText.append(tokens)
		return tokenizedText
