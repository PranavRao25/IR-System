from A1.sentenceSegmentation import SentenceSegmentation
from A1.tokenization import Tokenization
from A1.inflectionReduction import InflectionReduction
from A1.stopwordRemoval import StopwordRemoval

import argparse
import json
import os

import matplotlib.pyplot as plt

from LSA import LSAInformationRetrieval
from ESA import ESAInformationRetrieval
from TfIDF import TFIdfInformationRetrieval
from evaluation import Evaluation


'''# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print("Unknown python version - input function not safe")

'''
class SearchEngine:

	def __init__(self, args):
		self.args = args
		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.inflectionReducer = InflectionReduction()
		self.stopwordRemover = StopwordRemoval()

		# Create output folder if not exists
		if not os.path.exists(self.args.out_folder):
			os.makedirs(self.args.out_folder)
			os.makedirs(os.path.join(self.args.out_folder, "preprocess"))

	def segmentSentences(self, text):
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)

	def tokenize(self, text):
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)

	def reduceInflection(self, text):
		return self.inflectionReducer.reduce(text)

	def removeStopwords(self, text):
		return self.stopwordRemover.fromList(text)
	
	def removeStopwordsData(self, text):
		return self.stopwordRemover.fromdata(text)



	def preprocessQueries(self, queries):

		segmentedQueries = []
		for query in queries:
			segmentedQueries.append(self.segmentSentences(query))

		with open(os.path.join(self.args.out_folder,"preprocess", "segmented_queries.txt"), 'w') as f:
			json.dump(segmentedQueries, f)

		tokenizedQueries = []
		for query in segmentedQueries:
			tokenizedQueries.append(self.tokenize(query))

		with open(os.path.join(self.args.out_folder,"preprocess", "tokenized_queries.txt"), 'w') as f:
			json.dump(tokenizedQueries, f)

		reducedQueries = []
		for query in tokenizedQueries:
			reducedQueries.append(self.reduceInflection(query))

		with open(os.path.join(self.args.out_folder,"preprocess", "reduced_queries.txt"), 'w') as f:
			json.dump(reducedQueries, f)

		stopwordRemovedQueries = []
		for query in reducedQueries:
			stopwordRemovedQueries.append(self.removeStopwords(query))

		with open(os.path.join(self.args.out_folder,"preprocess", "stopword_removed_queries.txt"), 'w') as f:
			json.dump(stopwordRemovedQueries, f)

		return stopwordRemovedQueries


	def preprocessDocs(self, docs):

		segmentedDocs = []
		for doc in docs:
			segmentedDocs.append(self.segmentSentences(doc))

		with open(os.path.join(self.args.out_folder,"preprocess", "segmented_docs.txt"), 'w') as f:
			json.dump(segmentedDocs, f)

		tokenizedDocs = []
		for doc in segmentedDocs:
			tokenizedDocs.append(self.tokenize(doc))

		with open(os.path.join(self.args.out_folder,"preprocess", "tokenized_docs.txt"), 'w') as f:
			json.dump(tokenizedDocs, f)

		reducedDocs = []
		for doc in tokenizedDocs:
			reducedDocs.append(self.reduceInflection(doc))

		with open(os.path.join(self.args.out_folder,"preprocess", "reduced_docs.txt"), 'w') as f:
			json.dump(reducedDocs, f)

		stopwordRemovedDocs = []
		for doc in reducedDocs:
			stopwordRemovedDocs.append(self.removeStopwords(doc))

		with open(os.path.join(self.args.out_folder,"preprocess", "stopword_removed_docs.txt"), 'w') as f:
			json.dump(stopwordRemovedDocs, f)

		return stopwordRemovedDocs


	def evaluateDataset(self, rank_type):

		query_path = os.path.join(self.args.dataset, "cran_queries.json")
		doc_path = os.path.join(self.args.dataset, "cran_docs.json")

		with open(query_path, 'r') as f:
			queries_json = json.load(f)

		queries = [item["query"] for item in queries_json]
		query_ids = [item["query number"] for item in queries_json]
		processed_queries = self.preprocessQueries(queries)

		with open(doc_path, 'r') as f:
			docs_json = json.load(f)

		docs = [item["body"] for item in docs_json]
		doc_ids = [item["id"] for item in docs_json]
		processed_docs = self.preprocessDocs(docs)

		if rank_type == "lsa":

			print("Building LSA Index...")
			lsa = LSAInformationRetrieval()
			lsa.buildIndex(processed_docs, doc_ids)

			print("Ranking Queries using LSA...")
			doc_IDs_ordered = lsa.rank(processed_queries)

		elif rank_type == "esa":
			
			print("Building ESA Index...")
			esa = ESAInformationRetrieval()
			esa.buildIndex(processed_docs, doc_ids)

			print("Ranking Queries using ESA...")
			doc_IDs_ordered = esa.rank(processed_queries)

		elif rank_type == "tfidf":
			
			print("Building TF-IDF Index...")
			tfidf = TFIdfInformationRetrieval()
			tfidf.buildIndex(processed_docs, doc_ids)

			print("Ranking Queries using TF-IDF...")
			doc_IDs_ordered = tfidf.rank(processed_queries)

		# elif rank_type == "constrastive":
			
		# 	print("Building Contrastive Index...")
		# 	contrastive = ContrastiveInformationRetrieval()
		# 	contrastive.buildIndex(processed_docs, doc_ids)

		# 	print("Ranking Queries using Contrastive...")
		# 	doc_IDs_ordered = contrastive.rank(processed_queries)

	

		# Evaluation
		print("Evaluating System...")
		qrels_path = os.path.join(self.args.dataset, "cran_qrels.json")
		with open(qrels_path, 'r') as f:
			qrels = json.load(f)

		evaluator = Evaluation()
		
		precisions, recalls, fscores, MAPs, nDCGs, MRRs = [], [], [], [], [], []
		for k in range(1, 11):
			precisions.append(evaluator.meanPrecision(doc_IDs_ordered, query_ids, qrels, k))
			recalls.append(evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k))
			fscores.append(evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k))
			MAPs.append(evaluator.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k))
			nDCGs.append(evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k))
			MRRs.append(evaluator.meanReciprocalRank(doc_IDs_ordered, query_ids, qrels, k))

		print("\nPerformance of different Models")
		print(f"Vector Space Model ({rank_type.upper()})")
		print("Precisions\tRecalls\tFscores\tMAPs\tnDCGs\tMRRs")
		for i in range(10):
			p = round(precisions[i], 4)
			r = round(recalls[i], 4)
			f = round(fscores[i], 4)
			m = round(MAPs[i], 4)
			n = round(nDCGs[i], 4)
			mrr = round(MRRs[i], 4)
			print(f"{p}\t{r}\t{f}\t{m}\t{n}\t{mrr}")

		
		ks = range(1, 11)
		plt.figure(figsize=(10, 6))
		plt.plot(ks, precisions, label='Precisions', marker='o')
		plt.plot(ks, recalls, label='Recalls', marker='o')
		plt.plot(ks, fscores, label='Fscores', marker='o')
		plt.plot(ks, MAPs, label='MAPs', marker='o')
		plt.plot(ks, nDCGs, label='nDCGs', marker='o')
		plt.plot(ks, MRRs, label='MRRs', marker='o')
		
		plt.title(f"Performance metrics for Vector Space Model using {rank_type.upper()} (k=1 to 10)")
		plt.xlabel("k")
		plt.ylabel("Score")
		plt.xticks(ks)
		plt.legend()
		plt.grid(True)
		
		plot_path = os.path.join(self.args.out_folder, f"eval_plot_{rank_type}.png")
		plt.savefig(plot_path)
		plt.close()
		print(f"\nPlot saved to {plot_path}")


	def handleCustomQuery(self):

		print("Enter query below")
		query = input()

		self.preprocessQueries([query])

		doc_path = os.path.join(self.args.dataset, "cran_docs.json")

		with open(doc_path, 'r') as f:
			docs_json = json.load(f)

		docs = [item["body"] for item in docs_json]
		self.preprocessDocs(docs)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='main.py')

	parser.add_argument('-dataset', default="cranfield",
						help="Path to dataset folder")

	parser.add_argument('-out_folder', default="outputs",
						help="Path to output folder")

	parser.add_argument('-segmenter', default="punkt",
	                    help="Sentence Segmenter Type [naive|punkt]")

	parser.add_argument('-tokenizer', default="ptb",
	                    help="Tokenizer Type [naive|ptb]")

	parser.add_argument('-custom', action="store_true",
						help="Take custom query as input")

	args = parser.parse_args()

	searchEngine = SearchEngine(args)

	if args.custom:
		searchEngine.handleCustomQuery()
	else:
		searchEngine.evaluateDataset("tfidf")
		searchEngine.evaluateDataset("lsa")
		searchEngine.evaluateDataset("esa")
