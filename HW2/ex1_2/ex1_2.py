from string_preprocessor import StringPreprocessor
from collections import defaultdict
import math
import csv 
import time

class InvertedIndex:
    def __init__(self):
        self.tf = defaultdict(lambda: {})
        self.idf = None
        self.tf_idf = None
        self.norms = None
        self.total_docs = 0       

    def add_document(self, doc_id, text):
        # separates words based on ' '
        terms = text.split()  

        # compute tf in the doc for each term
        for term in terms:
            self.tf[term][doc_id] = 1 if doc_id not in self.tf[term] else self.tf[term][doc_id] + 1

        # update document counter
        self.total_docs += 1
    
    def store_inverted_index(self, file):
        with open(file, "w", encoding="utf-8") as f:
            for term in self.tf.keys():
                row = f"{term}: "
                for doc in self.tf[term]:
                    row += f"({doc}, {self.tf[term][doc]}); "

                f.write(row[:-2] + "\n")

    def calculate_idf(self):
        # calculate inverse document frequency for terms
        self.idf = {}
        for term in self.tf.keys():
            # compute the number of documents containing the term
            doc_count = len(self.tf[term])
            self.idf[term] = math.log10(self.total_docs/doc_count)
        
    def calculate_tf_idf(self):
        # calculate tf_idf and the norms of the tf_idf values for each document (will be used for cosine similarity)
        self.tf_idf={}
        self.norms=defaultdict(float)

        for term in self.tf.keys():
            for doc in self.tf[term]:
                # compute tf_idf for the term in the doc
                tf_idf = self.tf[term][doc] * self.idf[term]
                self.tf_idf[(term, doc)] = tf_idf
                # cumulate the sum of squared tf_idf values for the norm
                self.norms[doc] += tf_idf ** 2
        for doc in self.norms:
            # compute the norm by computing the squared root of the sum of squared tf_idf values
            self.norms[doc] = math.sqrt(self.norms[doc])

def fill_index(file, string_preprocessor=StringPreprocessor(lang="italian")):
    # open the file containing the products
    with open(file, mode='r', encoding='utf-8') as f:
        index = InvertedIndex()

        # access the file as a csv
        reader = csv.reader(f, delimiter='\t')
        # skip the header
        header = next(reader)  
        # fill the index (computes tf)
        products=[]
        for i, row in enumerate(reader):
            product_text = row[0]  
            products.append(row)
            product_text = string_preprocessor.preprocess(product_text)
            index.add_document(i, product_text)
    
        index.store_inverted_index("inverted_index.txt")
        # compute idf as well
        index.calculate_idf()
        # finally, compute tf_idf
        index.calculate_tf_idf()

    return index, products, header

class SearchEngine:
    def __init__(self, inverted_index=None):
        if inverted_index is None:
            inverted_index = self.import_from_file("inverted_index.txt")
        
        self.idf = inverted_index.idf
        self.tf_idf = inverted_index.tf_idf
        self.norms = inverted_index.norms

    def import_from_file(self, file):
            tf = {}
            num_docs = 0
            with open(file, "r", encoding="utf-8") as f:

                for line in f:
                    # Split line into key (word) and value (document list)
                    term, docs = line.strip().split(": ")
                    tf[term] = {}
                    # Process the docs to extract doc_id and tf
                    docs = docs.split("; ") 
                    
                    for doc in docs:
                        doc_id, term_freq = doc.strip("()").split(", ")
                        doc_id = int(doc_id) 
                        term_freq = int(term_freq)
                        num_docs = max(num_docs, doc_id+1)
                        tf[term][doc_id] = term_freq
            #recreate the inverted index using the number of docs and the tf dictionary
            index = InvertedIndex()
            index.tf = tf
            index.total_docs = num_docs
            index.calculate_idf()
            index.calculate_tf_idf()
            return index

    def search(self, query, top_k = 20, string_preprocessor=StringPreprocessor(lang="italian")):
        query = string_preprocessor.preprocess(query)
        query = query.split()
        query_tf = defaultdict(int)
        query_idf = defaultdict(float)
        candidates = {}

        # avoid repeating words which is useless
        words = set(query)
        # compute tf for terms in the query
        for word in words:
            query_tf[word] = query.count(word)
                
            for term, doc in self.tf_idf.keys():
                if term == word:
                    # recover idf values for words in the query
                    query_idf[word] = self.idf[word] 
                    # identify candidates document: they contain at least one word from the query vector, store their tf_idf
                    if doc not in candidates:
                        candidates[doc] = {word: self.tf_idf[(word, doc)]}
                    else:
                        candidates[doc][word] = self.tf_idf[(word, doc)]

        # each word from the query that does not appear in the candidate as tf_idf = 0 (tf is 0)
        for doc in candidates:    
            for word in words:
                if word not in candidates[doc]:
                    candidates[doc][word] = 0
        
        scores = {}
        # compute the cosine similarity
        for doc in candidates:
            query_tf_idf_scores = []
            dot_product = 0
            for word in candidates[doc]:
                # compute the tf_idf of the word in the query (what if word not in codebook?)
                query_tf_idf = query_tf[word] * query_idf[word]
                # add the score to a list to later compute the norm
                query_tf_idf_scores.append(query_tf_idf)
                # recover the tf_idf value of the word in the document
                doc_tf_idf = candidates[doc][word]
                # compute incrementally the dot product (sum of the product of tf_idf values of each word)
                dot_product += query_tf_idf * doc_tf_idf

            query_norm = math.sqrt(sum([tf_idf ** 2 for tf_idf in query_tf_idf_scores]))
            # normalize the dot product to compute the cosine similarity
            scores[doc] = dot_product / (query_norm * self.norms[doc])

        # sort the candidates by their cosine similarity and select the top_k candidates
        res = list(sorted(scores.items(), key=lambda item: item[1], reverse = True))[:top_k]
        return res

    def search_results(self, top_prods, products, header=None):
        results = {}
        for p in top_prods:
            doc_id, cosine_similarity = p
            # associate to the doc_id the corresponding product information
            prod_info = products[doc_id]
            results[doc_id] = {"Cosine Similarity": cosine_similarity} 

            # construct artificial header
            if header is None:
                header = ["Field " + str(i+1) for i in range(len(prod_info))]
            for i, p_info in enumerate(prod_info):
                results[doc_id][header[i]] = p_info                
                                
        return results



index, products, header = fill_index("products.tsv")
se = SearchEngine(index)

queries = ["Lenovo", "Mouse wireless", "Laptop ACER ssd 256 GB 15.6 pollici ram 16 gb intel i7 windows 11", "NOTODD PC Portatile Laptop Win11 12GB 512GB 1TB SSD Espansione, Notebook 16 Pollici Celeron N5095 (fino a 2.9Ghz) 丨Ventola di Raffreddamento丨5G WIFI丨1920 * 1200 2K Schermo Doppio- Viola"]
elapsed_times = 0
for q in queries:    
    print("*"*100)
    print(f"Query: {q}")
    start_time = time.time()
    top_docs = se.search(q)
    for i, (r, e) in enumerate(se.search_results(top_docs, products, header).items()):
        print(f"Ranked {i+1} document ID: {r}")
        for k, v in e.items():
            print(f"{k}: {v}")
        print('-'*100)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time}")
    elapsed_times += elapsed_time
print(f"Average elapsed time per query: {elapsed_times/len(queries)}")
