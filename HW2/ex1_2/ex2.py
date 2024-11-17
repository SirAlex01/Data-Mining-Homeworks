from pyspark import SparkContext
from string_preprocessor import StringPreprocessor
from pyspark.sql import SparkSession
import math
import time

def store_inverted_index(index, file):
    index = index.map(lambda line: (line[0], [line[1]]))
    index = index.reduceByKey(lambda docs1, docs2: docs1+docs2)

    with open(file, "w", encoding="utf-8") as file:
        for word, doc_tf_list in index.collect():  
            row = f"{word}: "
            for doc, tf in doc_tf_list: 
                row += f"({doc}, {tf}); "
            file.write(row[:-2] + "\n")

def import_inverted_index(file, sc):
    # Initialize an empty list to store the data for creating the RDD
    inverted_index_data = []

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            # Split the line into term and its associated document list
            term, docs = line.strip().split(": ")
            docs = docs.split("; ")

            for doc in docs:
                doc_id, term_freq = doc.strip("()").split(", ")
                doc_id = int(doc_id)
                term_freq = int(term_freq)
                inverted_index_data.append((term, (doc_id, term_freq)))

    # Create an RDD from the list of tuples
    inverted_index_rdd = sc.parallelize(inverted_index_data)

    # Return the RDD
    return inverted_index_rdd

# input: rdd with columns (document, doc_id)
# output: codebook with columns: (doc_id, word, idf, tf_idf, norm)
def create_codebook(documents):
    # begin creating the inverted index: isolate the words from the description and prepare for the count (TF) 
    # flatMap flats the lists of words from the same document
    inverted_index = documents.flatMap(lambda line: [((line[1] ,word),1) for word in line[0].split()])
    # reduce the tuples with the same pair (doc_id, word) to compute tf
    inverted_index = inverted_index.reduceByKey(lambda line1, line2 : line1+line2)
    # reformat the index into (word, (doc_id, tf))
    inverted_index = inverted_index.map(lambda line: (line[0][1], (line[0][0], line[1])))

    index_file = "inverted_index_spark.txt"
    store_inverted_index(inverted_index, index_file)
    # check that the import works
    #print(set(inverted_index.collect()) == set(import_inverted_index(index_file, sc).collect()))

    df = inverted_index.map(lambda line: (line[0], 1))
    # compute document frequency
    df = df.reduceByKey(lambda docs1,docs2 : docs1+docs2) 
    total_docs = documents.count()
    idf = df.map(lambda line: (line[0], math.log10(total_docs/line[1])))

    # after the join we have (word, ((doc_id, tf), idf)))
    tf_idf = inverted_index.join(idf)
    # reformat into (doc_id, (word, idf, tf_idf))
    tf_idf = tf_idf.map(lambda line: (line[1][0][0], (line[0], line[1][1], line[1][1]*line[1][0][1])))

    # isolate (doc_id, tf_idf) to compute the norm
    tf_idf_norms = tf_idf.map(lambda line: (line[0], line[1][-1]))
    # compute the norm: square all tf_idf values, sum them in the reduce and compute the squared root with a final map
    tf_idf_norms = tf_idf_norms.map(lambda line: (line[0], line[1]**2))
    tf_idf_norms = tf_idf_norms.reduceByKey(lambda val1, val2: val1+val2)
    tf_idf_norms = tf_idf_norms.map(lambda line: (line[0], math.sqrt(line[1])))

    #join the norms to the idf and tf_idf values
    tf_idf = tf_idf.join(tf_idf_norms)
    #produce the codebook: doc_id, word, idf, tf_idf, norm
    codebook = tf_idf.map(lambda line: (line[0], line[1][0][0], line[1][0][1], line[1][0][2], line[1][1]))
    return codebook

def search(query, codebook, top_k = 20):
    query = query.split()

    # extract the documents with a word present in the query
    filtered_codebook = codebook.filter(lambda line: line[1] in q)
    
    # compute tf of words in the query
    query_tf = sc.parallelize(query).map(lambda word: (word, 1))
    query_tf = query_tf.reduceByKey(lambda count1, count2: count1+count2)

    # join with the codebook words for idf values (also removes words not present in the codebook)
    idf_values = filtered_codebook.map(lambda line: (line[1], line[2])).distinct()
    # word, (tf, idf)
    query_tf_idf = query_tf.join(idf_values)
    # word, tf_idf
    query_tf_idf = query_tf_idf.map(lambda line: (line[0], line[1][0]*line[1][1]))

    # compute the norm of tf_idf values    
    query_norm = query_tf_idf.map(lambda line: line[1]**2)
    query_norm = math.sqrt(query_norm.reduce(lambda w1, w2: w1+w2))
    
    # to compute the dot_product, we need to join the non-zero dimensions in both the query and the documents
    # we get ready for the join extracting from the codebook word, (doc_id, tf_idf)
    tf_idf_doc = filtered_codebook.map(lambda line: (line[1], (line[0], line[-2])))
    # word, ((doc_id, tf_idf), query_tf_idf)
    combined_query_docs = tf_idf_doc.join(query_tf_idf)

    # doc_id, tf_idf*query_tf_idf
    dot_product = combined_query_docs.map(lambda line: (line[1][0][0], line[1][0][1] * line[1][1]))
    # compute the dot_product
    dot_product = dot_product.reduceByKey(lambda dim1, dim2: dim1+dim2)

    doc_norms = filtered_codebook.map(lambda line: (line[0], line[-1])).distinct()
    # doc_id, (dot_product, doc_norm)
    cosine_similarity = dot_product.join(doc_norms)
    # compute the cosine similarity dividing by the norms 
    cosine_similarity = cosine_similarity.map(lambda line: (line[0], line[1][0]/(line[1][1]*query_norm)))

    # sort results by similarity score in descending order
    results = cosine_similarity.sortBy(lambda line: line[1], ascending=False).zipWithIndex()
    # extract the top_k and preserve the results in an rdd
    top_k_rdd = results.filter(lambda line: line[1] < top_k).map(lambda line: line[0])
    return top_k_rdd

def search_results(top_prods, products, header=None):
    # swap prod_list and doc_id
    products = products.map(lambda line: (line[1], line[0]))

    # doc_id, (cosine_similarity, prod_list)
    results = top_prods.join(products)

    
    # flatten the results rdd and extract the items in the prod_list (extracting products info)
    results = results.map(lambda line: (line[0], line[1][0], *line[1][1]))
    results = results.sortBy(lambda line: line[1], ascending=False)

    # I use a dataframe to visualize the result
    spark = SparkSession(sc)
    if header is not None:
        # add doc_id and cosine similarity to the provided header
        full_header = ["Document ID", "Cosine Similarity"] + header
        results_df = spark.createDataFrame(results, schema=full_header)
    else:
        results_df = spark.createDataFrame(results)
    
    return results_df

# initialize SparkContext
sc = SparkContext('local[*]')
# add to the SparkContext the file in order to use the class
sc.addFile("string_preprocessor.py")

# get the products from the tsv file
products = sc.textFile("products.tsv")

# take first line (header needs to be removed)
header = products.first()
# filter out the header from the products
products = products.filter(lambda line: line != header).cache()  # Cache to ensure removal consistency
# split the lines of the tsv file, add the index of the documents (the products)
products = products.map(lambda line: line.split("\t")).zipWithIndex()
header = header.split("\t")

string_preprocessor = StringPreprocessor(lang="italian")
# isolate description and doc_index, preprocess the description
desc_with_index = products.map(lambda line: (string_preprocessor.preprocess(line[0][0]), line[1]))

codebook = create_codebook(desc_with_index)

queries = ["Lenovo", "Mouse wireless", "Laptop ACER ssd 256 GB 15.6 pollici ram 16 gb intel i7 windows 11", "NOTODD PC Portatile Laptop Win11 12GB 512GB 1TB SSD Espansione, Notebook 16 Pollici Celeron N5095 (fino a 2.9Ghz) 丨Ventola di Raffreddamento丨5G WIFI丨1920 * 1200 2K Schermo Doppio- Viola"]
elapsed_times = 0
for i, q in enumerate(queries):
    print('*'*100)
    print(f"Query: {q}")
    start_time = time.time()
    q = string_preprocessor.preprocess(q)
    top_prods = search(q, codebook)
    res = search_results(top_prods, products, header)
    res.show()
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time}")
    elapsed_times += elapsed_time
    res.toPandas().to_csv(f"query_{i+1}_results.csv")

print(f"Average elapsed time per query: {elapsed_times/len(queries)}")