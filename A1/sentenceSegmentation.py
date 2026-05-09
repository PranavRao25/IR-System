from A1.util import *

# Add your import statements here
import re
from nltk.tokenize import sent_tokenize

nltk.download('punkt_tab')

class SentenceSegmentation():

	def __init__(self):
		# Load spaCy model (students may use this if needed)
		self.nlp = spacy.load("en_core_web_sm")

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = []

		# Fill in code here
		sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
		for sent in sentences:
			sent = sent.strip()
			if sent:
				segmentedText.append(sent)
		return segmentedText


	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		segmentedText = sent_tokenize(text)
		
		return segmentedText


	def spacySegmenter(self, text):
		"""
		Sentence Segmentation using spaCy

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = []
		doc = self.nlp(text)
		for sent in doc.sents:
			segmentedText.append(sent.text)

		return segmentedText
