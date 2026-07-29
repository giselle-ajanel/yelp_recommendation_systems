import time
import sys
from pyspark import SparkContext
# from itertools import combinations

sc = SparkContext.getOrCreate()
start_time = time.time()

input_file = sys.argv[1]
output_file = sys.argv[2]

# load csv file
yelp_train = sc.textFile(input_file)
header = yelp_train.first()
remove_header = yelp_train.filter(lambda row: row != header)
split_rows = remove_header.map(lambda row: row.split(","))


# jaccard similarity
def jaccard_sim(business1, business2):
    intersection = len(business1 & business2)
    union = len(business1 | business2)
    return intersection / union


users = split_rows.map(lambda row: row[0]).distinct().collect()

# creating sets
ids_to_indices = {}
i = 0
for ids in users:
    ids_to_indices[ids] = i
    i += 1


# min hashing
def min_hash_businesses(row):
    user = row[0]
    business = row[1]
    user_set = ids_to_indices[user]
    return business, {user_set}

def prime(num):
    if num < 2:
        return False
    ind = 2
    while ind * ind <= num:
        if num % ind == 0:
            return False
        ind += 1
    return True

def next_prime(num):
    prime_num = num + 1
    while not prime(prime_num):
        prime_num += 1
    return prime_num


mapper = split_rows.map(min_hash_businesses)
combine_businesses = mapper.reduceByKey(
    lambda set1, set2: set1 | set2)


m = len(users)
p = next_prime(m)

num_hashes = 50
parameters = []
i = 0

while i < num_hashes:
    x = i + 1
    y = i + 2
    parameters.append((x, y))
    i += 1

def hash_users(pair, user_set):
    a = pair[0]
    b = pair[1]

    hashed = map(lambda x: ((a * x + b) % p) % m, user_set)
    return hashed

def minimum_hash(pair, user_set):
    hashed = hash_users(pair, user_set)
    return min(hashed)

def signature_funct(row):
    business_id = row[0]
    user_set = row[1]
    signature = list(map(lambda pair: minimum_hash(pair, user_set), parameters))

    return business_id, signature


# would be helpful for fast lookups
business_dict = combine_businesses.collectAsMap()

signatures = combine_businesses.map(signature_funct)

# locality sensitive hashing
# since hashes is 50; b*r=50
b = 25
r = 2

def split_business_sig(row):
    business_id = row[0]
    signature = row[1]

    # initializing
    bands = []
    for num_bands in range(0, b):
        initial_bands = num_bands * r
        band_ends = initial_bands + r
        band_section = tuple(signature[initial_bands:band_ends])
        bands.append(((num_bands, band_section), business_id))
    return bands


# one business -> many bands
many_bands = signatures.flatMap(split_business_sig)
buckets = many_bands.groupByKey().mapValues(lambda row: list(row))

# we create candidate pairs
def candidate_pairs(row):
    businesses = sorted(row[1])
    pairs = []

    i = 0
    while i < len(businesses):
        j = i + 1
        while j < len(businesses):
            pairs.append((businesses[i], businesses[j]))
            j += 1
        i += 1

    return pairs


candidate_pair_groups = buckets.flatMap(candidate_pairs).distinct()

# jaccard computation
def true_jaccard(pair):
    business1 = pair[0]
    business2 = pair[1]

    set1 = business_dict[business1]
    set2 = business_dict[business2]
    similarity = jaccard_sim(set1, set2)

    return business1, business2, similarity


jaccard_results = candidate_pair_groups.map(true_jaccard)
true_pairs = jaccard_results.filter(lambda row: row[2] >= 0.5)


# sorting lexicographically
results = true_pairs.sortBy(lambda x: (x[0], x[1])).collect()

# output
final_all = "business_id_1,business_id_2,similarity\n"

for x in results:
    business1 = x[0]
    business2 = x[1]
    similarity = x[2]
    final_all += business1 + "," + business2 + "," + str(similarity) + "\n"

output_file_write = open(output_file, "w")
output_file_write.write(final_all)
output_file_write.close()

end_time = time.time()
custom_exe_time = end_time - start_time
print("Duration:", custom_exe_time)
