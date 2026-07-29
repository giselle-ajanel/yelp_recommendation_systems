import json
import time
import sys
from xgboost import XGBRegressor
from pyspark import SparkContext

sc = SparkContext.getOrCreate()
start_time = time.time()

folder_path = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

train_file = folder_path + "/yelp_train.csv"
user_file = folder_path + "/user.json"
business_file = folder_path + "/business.json"

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


# read json files
user_folder_file = sc.textFile(user_file)
user_json = user_folder_file.map(lambda x: json.loads(x))
extract_user_features = user_json.map(lambda x: (x["user_id"],
                                      (x.get("review_count", 0),
                                       x.get("useful", 0),
                                       x.get("funny", 0),
                                       x.get("cool", 0),
                                       x.get("fans", 0),
                                       x.get("average_stars", 0.0)
                                       )))

business_folder_file = sc.textFile(business_file)
business_json = business_folder_file.map(lambda x: json.loads(x))
extract_business_features = business_json.map(lambda x: (x["business_id"],
                                      (x.get("stars", 0.0),
                                       x.get("review_count", 0),
                                       x.get("latitude", 0.0),
                                       x.get("longitude", 0.0),
                                       x.get("is_open", 0.0)
                                       )))

# creating dictionaries
user_file_dict = extract_user_features.collectAsMap()
business_file_dict = extract_business_features.collectAsMap()

# .............
def pairs_to_dict(pairs):
    result = {}
    for b, r in pairs:
        result[b] = r
    return result

# def normalization(num):
#     values = list(num)
#     return sum(values) / len(values)


user_dict = create_tuples.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(pairs_to_dict).collectAsMap()
business_dict = create_tuples.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(pairs_to_dict).collectAsMap()


user_pairs = create_tuples.map(lambda x: (x[0], (x[2], 1)))
def add_users(x, y):
    sum_rating = x[0] + y[0]
    count = x[1] + y[1]
    return sum_rating, count


user_totals = user_pairs.reduceByKey(add_users)
user_totals_dict = user_totals.collectAsMap()

def calculate_avg(rating):
    sum_rating = rating[0]
    count = rating[1]
    return sum_rating / count


user_avg = user_totals.mapValues(calculate_avg)
user_avg_dict = user_avg.collectAsMap()


business_pairs = create_tuples.map(lambda x: (x[1], (x[2], 1)))
business_totals = business_pairs.reduceByKey(add_users)
business_totals_dict = business_totals.collectAsMap()
business_average = business_totals.mapValues(calculate_avg)
business_avg_dict = business_average.collectAsMap()

# reusable
all_avg = create_tuples.map(lambda x: x[2]).mean()

# was using too many 0s helped with similarity on business avgs
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


user_threshold = 80 # 40 for task 2_2
business_threshold = 80 # 40 for task 2_2; before 70

def features_xgboost(user, business):
    u_avg_raw = user_avg_dict.get(user, all_avg)
    u_count = user_totals_dict.get(user, (0, 0))[1]

    b_avg_raw = business_avg_dict.get(business, all_avg)
    b_count = business_totals_dict.get(business, (0, 0))[1]


    u_smoothed = ((u_avg_raw * u_count) + (user_threshold * all_avg)) / (u_count + user_threshold)
    b_smoothed = ((b_avg_raw * b_count) + (business_threshold * all_avg)) / (b_count + business_threshold)

    u_bias = u_smoothed - all_avg
    b_bias = b_smoothed - all_avg

    diff_bias = u_smoothed - b_smoothed
    

    user_features = user_file_dict.get(user, (0, 0, 0, 0, 0, all_avg))
    business_features = business_file_dict.get(business, (all_avg, 0, 0.0, 0.0, 0.0))

    return [
        u_smoothed,
        b_smoothed,
        u_bias,
        b_bias,
        diff_bias,
        user_features[5],
        business_features[0],
        user_features[0],
        business_features[1],
        user_features[4],
        business_features[2],
        business_features[3],
        business_features[4]
    ]


# predicting train file
def cf_prediction(user, business):
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


# train features
train_data = create_tuples.collect()

x_train = []
y_train = []

# issues with looping over an empty list :(
for feature in train_data:
    user_feature = feature[0]
    business_feature = feature[1]
    rating = feature[2]

    all_features = features_xgboost(user_feature, business_feature)
    x_train.append(all_features)
    y_train.append(rating)


# test features
test_data = test_create_tuples.collect()

x_test = []
test_prediction = []

for pair in test_data:
    user_testing = pair[0]
    business_testing = pair[1]
    test_prediction.append((user_testing, business_testing))

    pair_features = features_xgboost(user_testing, business_testing)
    x_test.append(pair_features)

xgb_regressor = XGBRegressor(
    max_depth=4,
    n_estimators=1000, # before 1000
    learning_rate=0.01, # before 0.01
    subsample=0.75, # before 0.7
    colsample_bytree=0.75, # before 0.7
    reg_lambda=5.0, # before 3.0
    n_jobs=1
)

xgb_regressor.fit(x_train, y_train)
testing = xgb_regressor.predict(x_test)


# adjusting for hybrid rec system
alpha = 0.1

results = []
for i in range(len(test_prediction)):
    user = test_prediction[i][0]
    business = test_prediction[i][1]

    model = testing[i]
    cf_preds = cf_prediction(user, business)

    if user not in user_dict:
        prediction_num = model
    elif business not in business_dict:
        prediction_num = model
    else:
        prediction_num = (alpha * cf_preds) + ((1 - alpha) * model)

    if prediction_num < 1.0:
        prediction_num = 1.0
    if prediction_num > 5.0:
        prediction_num = 5.0
    results.append((user, business, prediction_num))


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