import csv
import hashlib
from collections import defaultdict
import math
import time

def hashFamily(i):
    resultSize = 8 # how many bytes we want back
    maxLen = 20 # how long can our i be (in decimal)
    salt = str(i).zfill(maxLen)[-maxLen:].encode()
    def hashMember(x):
        x = str(x).encode()
        return hashlib.sha1(x + salt).digest()[-resultSize:]
    return hashMember

class Shingling():
    def __init__(self, k = 10):
        self.k = k

    def shingleDocument(self, doc):
        # if the documents doesn't even have k chars, produce a single shingle with the whole document
        if len(doc) < self.k:
            return set(doc)
        
        shingles = set()        
        for i in range(len(doc)):
            # split the document into shingles
            shingles.add(doc[i: i+self.k])
            # stop if the length of the document is reached
            if i + self.k == len(doc):
                break

        return shingles

    def hashShingles(self, shingles, hashFn):
        # produce the hash of the shingles
        return [hashFn(s) for s in shingles]

class MinHashing:
    def __init__(self, num_hashes, hashFamily):
        self.num_hashes = num_hashes
        self.hashFamily = hashFamily
    
    def generateSignature(self, objects):
        signature = []
        for i in range(self.num_hashes):
            # the i-th hash fucntion will provide a random permutation of all objects due to the intrinsic randomicity of 
            # cryptographic hash functions which will shuffle the objects from the domain in the codomain of the hash function
            hashFn = self.hashFamily(i)
            # if the minimum value is the same, the first object that appears in the permutation in both sets is the same
            min_hash = min([hashFn(o) for o in objects])

            # check randomicity of hash and correspondance between integers and bytes
            #print([hashFn(o) for o in objects].index(min_hash), [int.from_bytes(hashFn(o)) for o in objects].index(int.from_bytes(min_hash)))
            signature.append(min_hash)
        
        return signature
    
class NearestNeighbors:
    @staticmethod
    def jaccardSimilarity(doc1, doc2):
        # compute jaccard similairty between two sets
        intersection = len(doc1 & doc2)
        union = len(doc1 | doc2)
        return intersection / union

    def findNearestNeighbors(self, shingles, threshold):
        n = len(shingles)
        results = set()
        # collect all pairs of documents whose shingles have an higher jaccard similarity than the threshold
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.jaccardSimilarity(shingles[i], shingles[j])
                if sim >= threshold:
                    results.add((i,j))

        return results

class LSH:
    def __init__(self, bands, rows_per_band):
        self.b = bands
        self.r = rows_per_band

    def findNearestNeighbors(self, signatures):
        # make sure that the division into rows and bands matches the length of minhashing signature
        assert len(signatures[0]) == self.b * self.r, f"rows of signatures {len(signatures[0])} should be equal to {self.b * self.r}"
        
        # the buckets will contain a dictionary for each band, which collects the documents having
        # the same values in that band
        # the band can be hashed by default as a tuple by python, so we don't need to reuse tha hash family
        buckets = [defaultdict(list) for _ in range(self.b)]
        
        for id, sign in enumerate(signatures):
            for b in range(self.b):
                # extract the bound of the b-th band
                start = b * self.r
                end = start + self.r
                # extract the band from the signature
                band = sign[start:end]
                # add the band to the dictionary of b-th bands and associate to it the documents with the same band 
                buckets[b][tuple(band)].append(id)
        
        candidate_pairs = set()
        for b in range(self.b):
            for bucket_docs in buckets[b].values():
                # only consider buckets with more than 1 document
                if len(bucket_docs) > 1:  
                    # sort the documents to produce unique pairs
                    bucket_docs = sorted(bucket_docs)
                    for i in range(len(bucket_docs)):
                        for j in range(i + 1, len(bucket_docs)):
                            # add the pairs of similar documents (having the same b-th band)
                            candidate_pairs.add((bucket_docs[i], bucket_docs[j]))
                            # check same b-th band
                            #print(signatures[bucket_docs[i]][b*self.r:(b+1)*self.r]==signatures[bucket_docs[j]][b*self.r:(b+1)*self.r])

        return candidate_pairs

                
s = 0.8

# choose the desired number of bands, the number of rows will be chosen accordingly
MODE = "choose_bands"       
b = 300

# alternatively: choose the desired number of rows per band
#MODE = "choose_rows" 
r = 25

if MODE == "choose_bands":
    # exploiting the fact that s is about (1/b)^(1/r)
    r = round(-math.log(b)/math.log(s))
elif MODE == 'choose_rows':
    # exploiting the fact that s is about (1/b)^(1/r)
    b = round(1/(s**r))
else:
    raise ValueError("Invalid MODE. Must be either 'choose_bands' or 'choose_rows'.")


eps = 0.05
print(f"LSH bands: {b}, rows per band: {r}")
print(f"The probability of detecting two documents x,y as near-duplicates s.t. J(x,y)>={s} is above {1-(1-s**r)**b}")
print(f"The probability of detecting two documents x,y as near-duplicates s.t. J(x,y)<={s-eps:.3f} is below {1-(1-(s-eps)**r)**b}")


sh = Shingling(k=10)
mh = MinHashing(r*b, hashFamily)

# open the file containing the products
with open("products.tsv", mode='r', encoding='utf-8') as f:
    # access the file as a csv
    reader = csv.reader(f, delimiter='\t')
    # skip the header
    header = next(reader)  

    products=[]
    signatures = []
    shingles_list = []
    hashFn = hashFamily(0)
    for i, row in enumerate(reader):
        product_text = row[0]  
        products.append(row)

        # shingle the current product description
        shingles = sh.shingleDocument(product_text)
        shingles_list.append(shingles)
        # hash the shingles
        hashed = sh.hashShingles(shingles, hashFn)
        # generate the minwise hashing signature of the hashed shingles
        sign = mh.generateSignature(hashed)
        signatures.append(sign)

lsh = LSH(b, r)
start = time.time()
candidates = lsh.findNearestNeighbors(signatures)
elapsed_lsh = time.time()-start

nn = NearestNeighbors()
start = time.time()
real = nn.findNearestNeighbors(shingles_list, threshold=s)
elapsed_nn = time.time() - start
print(f"LSH took {elapsed_lsh} seconds")
print(f"Nearest neighbors took {elapsed_nn} seconds")

print(f"LSH found {len(candidates)} near-duplicates")
print(f"The actual number of near-duplicates is {len(real)}")
print(f"Near-duplicates in common: {len(candidates & real)}")

print("Wrong pairs detected by LSH:")
for fp in candidates - real:
    print(f"Actual similarity of {fp} is {NearestNeighbors.jaccardSimilarity(shingles_list[fp[0]], shingles_list[fp[1]])}")

print("Correct pairs not detected by LSH:")
for fn in real - candidates:
    print(f"Actual similarity of {fn} is {NearestNeighbors.jaccardSimilarity(shingles_list[fn[0]], shingles_list[fn[1]])}")
