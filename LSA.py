from A1.util import *
from collections import Counter
import numpy as np


class LSAInformationRetrieval():

	def __init__(self, num_components = 100):
		self.num_components = num_components
		self.doc_ids = []
		self.doc_id_to_index = {}
		self.vocabulary = {}
		self.idf = None
		self.doc_vectors = None
		self.doc_norms = None
		self.left_singular_vectors = None
		self.inverse_singular_values = None

	def _flatten_tokens(self, text):
		"""
		Flattens a document/query from sentence-token form to a single token list.
		"""

		return [token for sentence in text for token in sentence]

	def _build_query_vector(self, query):
		"""
		Builds a TF-IDF vector for a query using the document vocabulary and IDF.
		"""

		query_tokens = self._flatten_tokens(query)
		query_vector = np.zeros(len(self.vocabulary), dtype = float)

		if not query_tokens:
			return query_vector

		token_counts = Counter(query_tokens)
		query_length = len(query_tokens)

		for token, count in token_counts.items():
			term_index = self.vocabulary.get(token)
			if term_index is None:
				continue

			tf = count / query_length
			query_vector[term_index] = tf * self.idf[term_index]

		return query_vector

	def buildIndex(self, docs, docIDs):
		"""
		Builds the latent semantic index for the given document collection.

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		self.doc_ids = list(docIDs)
		self.doc_id_to_index = {
			doc_id: index for index, doc_id in enumerate(self.doc_ids)
		}

		flattened_docs = [self._flatten_tokens(doc) for doc in docs]
		doc_term_counts = [Counter(doc_tokens) for doc_tokens in flattened_docs]

		document_frequency = {}
		for term_counts in doc_term_counts:
			for term in term_counts.keys():
				document_frequency[term] = document_frequency.get(term, 0) + 1

		self.vocabulary = {
			term: index for index, term in enumerate(sorted(document_frequency.keys()))
		}

		num_terms = len(self.vocabulary)
		num_docs = len(self.doc_ids)

		if num_terms == 0 or num_docs == 0:
			self.idf = np.array([], dtype = float)
			self.doc_vectors = np.zeros((0, 0), dtype = float)
			self.doc_norms = np.array([], dtype = float)
			self.left_singular_vectors = np.zeros((0, 0), dtype = float)
			self.inverse_singular_values = np.array([], dtype = float)
			return

		term_document_matrix = np.zeros((num_terms, num_docs), dtype = float)
		self.idf = np.zeros(num_terms, dtype = float)

		for term, term_index in self.vocabulary.items():
			df = document_frequency[term]
			self.idf[term_index] = math.log((num_docs + 1) / (df + 1)) + 1

		for doc_index, term_counts in enumerate(doc_term_counts):
			doc_length = sum(term_counts.values())
			if doc_length == 0:
				continue

			for term, count in term_counts.items():
				term_index = self.vocabulary[term]
				tf = count / doc_length
				term_document_matrix[term_index, doc_index] = tf * self.idf[term_index]

		max_components = min(self.num_components, num_terms, num_docs)
		if max_components == 0:
			self.doc_vectors = np.zeros((num_docs, 0), dtype = float)
			self.doc_norms = np.zeros(num_docs, dtype = float)
			self.left_singular_vectors = np.zeros((num_terms, 0), dtype = float)
			self.inverse_singular_values = np.array([], dtype = float)
			return

		# Use the smaller document-document matrix to recover the top latent space.
		document_correlation = np.dot(term_document_matrix.T, term_document_matrix)
		eigenvalues, eigenvectors = np.linalg.eigh(document_correlation)

		sorted_indices = np.argsort(eigenvalues)[::-1]
		eigenvalues = eigenvalues[sorted_indices]
		eigenvectors = eigenvectors[:, sorted_indices]

		positive_mask = eigenvalues > 1e-10
		eigenvalues = eigenvalues[positive_mask]
		eigenvectors = eigenvectors[:, positive_mask]

		if eigenvalues.size == 0:
			self.doc_vectors = np.zeros((num_docs, 0), dtype = float)
			self.doc_norms = np.zeros(num_docs, dtype = float)
			self.left_singular_vectors = np.zeros((num_terms, 0), dtype = float)
			self.inverse_singular_values = np.array([], dtype = float)
			return

		# Calculate num_components to explain at least 95% of the variance
		total_variance = np.sum(eigenvalues)
		cumulative_variance = np.cumsum(eigenvalues)
		target_variance = 0.95 * total_variance
		
		num_components = np.argmax(cumulative_variance >= target_variance) + 1
		num_components = min(num_components, eigenvalues.size)

		singular_values = np.sqrt(eigenvalues[:num_components])
		right_singular_vectors = eigenvectors[:, :num_components]

		self.left_singular_vectors = np.dot(
			term_document_matrix,
			right_singular_vectors / singular_values
		)
		self.inverse_singular_values = 1.0 / singular_values

		# Each row is a document vector in the reduced latent space.
		self.doc_vectors = right_singular_vectors * singular_values
		self.doc_norms = np.linalg.norm(self.doc_vectors, axis = 1)

	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query.

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		if not self.doc_ids:
			return doc_IDs_ordered

		for query in queries:
			query_vector = self._build_query_vector(query)

			if self.left_singular_vectors.size == 0 or self.inverse_singular_values.size == 0:
				doc_IDs_ordered.append(list(self.doc_ids))
				continue

			latent_query = np.dot(query_vector, self.left_singular_vectors)
			latent_query = latent_query * self.inverse_singular_values

			query_norm = np.linalg.norm(latent_query)
			if query_norm == 0:
				sorted_doc_ids = sorted(self.doc_ids)
				doc_IDs_ordered.append(sorted_doc_ids)
				continue

			scores = np.dot(self.doc_vectors, latent_query)
			denominator = self.doc_norms * query_norm
			similarities = np.divide(
				scores,
				denominator,
				out = np.zeros_like(scores),
				where = denominator > 0
			)

			sorted_indices = sorted(
				range(len(self.doc_ids)),
				key = lambda index: (-similarities[index], self.doc_ids[index])
			)
			doc_IDs_ordered.append([self.doc_ids[index] for index in sorted_indices])

		return doc_IDs_ordered
