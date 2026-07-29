import time
import sys
from pyspark import SparkContext

sc = SparkContext.getOrCreate()
start_time = time.time()

train_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

# read train files
yelp_train = sc.textFile(train_file)
header = yelp_train.first()
remove_header = yelp_train.filter(lambda x: x != header)
split_rows = remove_header.map(lambda x: x.split(","))
create_tuples = split_rows.map(lambda x: (x[0], x[1], float(x[2])))

# read test files
yelp_test = sc.textFile(test_file)
test_header = yelp_test.first()
remove_test_header = yelp_test.filter(lambda x: x != test_header)
test_split_rows = remove_test_header.map(lambda x: x.split(","))
test_create_tuples = test_split_rows.map(lambda x: (x[0], x[1]))

# function that will be reusable
def pairs_to_dict(pairs):
    result = {}
    for b, r in pairs:
        result[b] = r
    return result

def normalization(num):
    values = list(num)
    return sum(values) / len(values)


# dictionary keyed by user
user_dict = create_tuples.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(pairs_to_dict).collectAsMap()
user_avg_dict = create_tuples.map(lambda x: (x[0], x[2])).groupByKey().mapValues(normalization).collectAsMap()

# dictionary keyed by business
business_dict = create_tuples.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(pairs_to_dict).collectAsMap()
business_avg_dict = create_tuples.map(lambda x: (x[1], x[2])).groupByKey().mapValues(normalization).collectAsMap()

# avg for all to help predict when data is missing
all_avg = create_tuples.map(lambda x: x[2]).mean()

# was using too many 0s before hand, this helped basing similarity on business avgs 
def business_avg_sim(business1, business2):
    business_avg = business_avg_dict.get(business1, all_avg)
    business_avg2 = business_avg_dict.get(business2, all_avg)
    return (5.0 - abs(business_avg - business_avg2)) / 5.0

# pearson similarity
def pearson_similarity(business1, business2):
    diction = {}
    users1 = business_dict.get(business1, diction)
    users2 = business_dict.get(business2, diction)

    users_intersect_at = set(users1.keys()) & set(users2.keys())

    if len(users_intersect_at) <= 1:
        return business_avg_sim(business1, business2)

    rating1 = []
    rating2 = []
    for user in users_intersect_at:
        rating1.append(users1[user])
        rating2.append(users2[user])

    # getting avg
    average1 = sum(rating1) / len(rating1)
    average2 = sum(rating2) / len(rating2)

    numerator = 0.0
    denominator1 = 0.0
    denominator2 = 0.0
    for i in range(len(rating1)):
        normalize1 = rating1[i] - average1
        normalize2 = rating2[i] - average2

        numerator += normalize1 * normalize2
        denominator1 += normalize1 * normalize1
        denominator2 += normalize2 * normalize2

    if denominator1 == 0.0 or denominator2 == 0.0:
        return business_avg_sim(business1, business2)

    return numerator / ((denominator1 * denominator2) ** 0.5)


# predicting train file
def prediction(user, business):
    cold_case = {}
    get_user = user_dict.get(user, cold_case)
    get_business = business_dict.get(business, cold_case)

    # cold start issue we need to resolve
    # if both are missing return all avg
    if get_user == cold_case and get_business == cold_case:
        return all_avg
    # if user is missing return business avg
    if get_user == cold_case:
        return business_avg_dict[business]

    # if business is missing return user avg
    if get_business == cold_case:
        return user_avg_dict[user]


    rated_items = list(get_user.items())

    # initializing
    i = 0
    numerator = 0.0
    denominator = 0.0
    while i < len(rated_items):
        business_rating_pair = rated_items[i]
        business_part = business_rating_pair[0]
        rating_part = business_rating_pair[1]

        if business_part != business:
            similarity = pearson_similarity(business, business_part)
            if similarity > 0:
                numerator += similarity * rating_part
                denominator += abs(similarity)
        i += 1

    if denominator == 0.0:
        preds = (user_avg_dict[user] + business_avg_dict[business]) / 2.0
    else:
        preds = numerator / denominator
    return preds


# predicting test file
test_predictions = test_create_tuples.map(lambda x: (x[0], x[1], prediction(x[0], x[1])))
results = test_predictions.collect()

# save
# output
final_all = "user_id,business_id,prediction\n"

for x in results:
    user_id = x[0]
    business_id = x[1]
    prediction_results = x[2]
    final_all += user_id + "," + business_id + "," + str(prediction_results) + "\n"

output_file_write = open(output_file, "w")
output_file_write.write(final_all)
output_file_write.close()

end_time = time.time()
custom_exe_time = end_time - start_time
print("Duration:", custom_exe_time)