from pyspark import SparkContext
from string_preprocessor import StringPreprocessor
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import time

def create_codebook(documents):
    # split the description into words (explode the words into rows)
    inverted_index = documents.withColumn("word", F.explode(F.split(F.col("processed_description"), " ")))
    inverted_index = inverted_index.select("doc_id", "word")
    inverted_index = inverted_index.groupBy("doc_id", "word").count().withColumnRenamed("count", "tf")

    df = inverted_index.groupBy("word").count().withColumnRenamed("count", "df").select("word","df")
    total_docs = documents.count()
    idf = df.withColumn("idf", (F.log10(total_docs / df["df"])))

    tf_idf = inverted_index.join(idf, on="word", how="inner")
    tf_idf = tf_idf.withColumn("tf_idf", tf_idf["tf"] * tf_idf["idf"])

    tf_idf_norms = tf_idf.withColumn("tf_idf_squared", tf_idf["tf_idf"] ** 2)
    tf_idf_norms = tf_idf_norms.groupBy("doc_id").sum("tf_idf_squared")
    tf_idf_norms = tf_idf_norms.withColumn("norm", F.sqrt(F.col("sum(tf_idf_squared)")))

    tf_idf = tf_idf.join(tf_idf_norms, on="doc_id", how="inner")
    codebook = tf_idf.select("doc_id", "word", "idf", "tf_idf", "norm")
    return codebook


def search(query, codebook, top_k=20):

    query = query.split()
    query = spark.createDataFrame([(word,) for word in query], schema=["query_word"])

    query_tf = query.groupBy("query_word").count().withColumnRenamed("count", "tf")

    filtered_codebook = codebook.join(query_tf, on=(codebook["word"] == query_tf["query_word"]), how="inner")

    idf_values = filtered_codebook.select("query_word", "idf").distinct()

    query_tf_idf = query_tf.join(idf_values, on="query_word", how="inner").withColumnRenamed("query_word", "word")
    query_tf_idf = query_tf_idf.withColumn("query_tf_idf", F.col("tf") * F.col("idf"))

    query_norm = query_tf_idf.withColumn("tf_idf_squared", F.col("query_tf_idf") ** 2)
    query_norm = query_norm.agg(F.sqrt(F.sum("tf_idf_squared")))
    # extract query_norm value from row object
    query_norm = query_norm.collect()[0][0]


    tf_idf_doc = filtered_codebook.select("doc_id", "word", "tf_idf")
    
    combined_query_docs = tf_idf_doc.join(query_tf_idf.select("word", "query_tf_idf"), on="word", how="inner")
    combined_query_docs = combined_query_docs.withColumn("dot_product_component", F.col("tf_idf") * F.col("query_tf_idf"))

    dot_product = combined_query_docs.groupBy("doc_id").sum("dot_product_component").withColumnRenamed("sum(dot_product_component)", "dot_product")

    doc_norms = filtered_codebook.select("doc_id", "norm").distinct()

    cosine_similarity = dot_product.join(doc_norms, on="doc_id", how="inner")
    cosine_similarity = cosine_similarity.withColumn("cosine_similarity", F.col("dot_product") / (F.col("norm") * query_norm))

    results = cosine_similarity.orderBy(F.col("cosine_similarity").desc()).limit(top_k)

    return results.select("doc_id", "cosine_similarity")



def search_results(top_k_results, products):

    results = top_k_results.join(products, on="doc_id", how="inner")

    results = results.select("doc_id", "cosine_similarity", *products.columns[:-1])

    results = results.orderBy(F.col("cosine_similarity").desc())

    return results.withColumnRenamed("doc_id", "Document ID").withColumnRenamed("cosine_similarity", "Cosine Similarity")


sc = SparkContext('local[*]')
sc.addFile("string_preprocessor.py")

# initialize spark session
spark = SparkSession.builder.appName("tf_idf").getOrCreate()

# load the TSV file into a DataFrame
products = spark.read.option("delimiter", "\t").csv("products.tsv", header=True)
products = products.withColumn("doc_id", F.monotonically_increasing_id())


# register the UDF for description preprocessing
preprocess_udf = udf(StringPreprocessor(lang="italian").preprocess, StringType())
desc_with_index = products.select("description", "doc_id").withColumn("processed_description", preprocess_udf(F.col("description")))

codebook = create_codebook(desc_with_index)

queries = ["Lenovo", "Mouse wireless", "Laptop ACER ssd 256 GB 15.6 pollici ram 16 gb intel i7 windows 11", "NOTODD PC Portatile Laptop Win11 12GB 512GB 1TB SSD Espansione, Notebook 16 Pollici Celeron N5095 (fino a 2.9Ghz) 丨Ventola di Raffreddamento丨5G WIFI丨1920 * 1200 2K Schermo Doppio- Viola"]
elapsed_times = 0
for q in queries:
    print('*'*100)
    print(f"Query: {q}")
    start_time = time.time()
    q = StringPreprocessor(lang="italian").preprocess(q)
    top_prods = search(q, codebook)
    res = search_results(top_prods, products)
    res.show()
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time}")
    elapsed_times += elapsed_time
    
print(f"Average elapsed time per query: {elapsed_times/len(queries)}")