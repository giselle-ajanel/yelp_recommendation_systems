"""
Method Description:
The methodology for this recommendation system was a hybrid one while also combing a model based approach through using XGBoost. This approach included using Pearson similarity, or item based CF, to predict Yelp star ratings. For user features, I included Bayesian smoothed average rating, review count, votes that are “useful,” “funny,” and “cool,” average stars, friend and elite status. For my business features, I used a smoothed average rating, review count, is_open, price range, category count, category flags for food, restaurants coffee, nightlife, fast food, Bars, and attributes on credit cards, delivery, good for kids, and reservations. For my XGBoost model, I hypertuned my model locally to the features that I used. My model used max_depth=4, n_estimators=1000, and learning_rate=0.01. I would like to say that I created a hybrid alpha that differed based on review count. For example for sparse users/businesses an alpha of 0.0 was used, for moderate users/businesses an alpha of 0.06 was used, for active users/businesses an alpha of 0.12 was used, and for highly active users/businesses an alpha of 0.14 was implemented. This shows how alpha gave significantly more weight to collaborative filtering when users/businesses had enough history. Finally, predictions were scaled to 1.0 to 5.0 to match Yelp ratings.


Error Distribution:
>=0 and <1: 101320                                                                                                                                      
>=1 and <2: 33286                                                                                                                                       
>=2 and <3: 6524                                                                                                                                        
>=3 and <4: 913                                                                                                                                         
>=4: 1 


RMSE: 0.9949330304926925   

Execution Time: 653.071123123169   

"""

import json
import time
import sys
import math
from xgboost import XGBRegressor
from pyspark import SparkContext

sc = SparkContext.getOrCreate()
start_time = time.time()

folder_path = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]


train_file = folder_path + "/yelp_train.csv"
user_file = folder_path + "/user.json"
business_file = folder_path + "/business.json"

# updates for competition.py
# review_train_file = folder_path + "/review_train.json"

# read train files
yelp_train = sc.textFile(train_file)
header = yelp_train.first()
remove_header = yelp_train.filter(lambda x: x != header)
split_rows = remove_header.map(lambda x: x.split(","))
create_tuples = split_rows.map(lambda x: (x[0], x[1], float(x[2])))

# read test files
yelp_test = sc.textFile(test_file_name)
test_header = yelp_test.first()
remove_test_header = yelp_test.filter(lambda x: x != test_header)
test_split_rows = remove_test_header.map(lambda x: x.split(","))
test_create_tuples = test_split_rows.map(lambda x: (x[0], x[1]))

def user_count(value):
    if value is None:
        return 0
    if value == "None":
        return 0
    if value == "":
        return 0

    value_string = str(value)
    count_split = value_string.split(",")
    splitting_count = len(count_split)
    return splitting_count


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

# good for turning boolean vals to nums
def boolean_translate(yelp_boolean):
    if yelp_boolean == "True":
        return 1
    if yelp_boolean == "False":
        return 0

    return 0

def price_range(attributes: dict):
    if attributes is None:
        return 0

    val = attributes.get("RestaurantsPriceRange2", "0")
    if val == "1":
        return 1
    elif val == "2":
        return 2
    elif val == "3":
        return 3
    elif val == "4":
        return 4

    return 0


def business_in_category(categories, word):
    if categories is None:
        return 0
    if word in categories:
        return 1
    return 0

def business_count_category(categories: str):
    if categories is None or categories == "":
        return 0
    return len(categories.split(","))


def attribute_val(attributes, key):
    if attributes is None:
        return 0
    val = attributes.get(key, "False")
    return boolean_translate(val)


# might add the wifi part but tbh don't know

business_folder_file = sc.textFile(business_file)
business_json = business_folder_file.map(lambda x: json.loads(x))
extract_business_features = business_json.map(lambda x: (x["business_id"],
                                                         (x.get("stars", 0.0),
                                                          x.get("review_count", 0),
                                                          x.get("is_open", 0.0),
                                                          price_range(x.get("attributes", {})),
                                                          business_count_category(x.get("categories", "")),


                                                          # x.get("latitude", 0.0),
                                                          # x.get("longitude", 0.0),
                                                          business_in_category(x.get("categories", ""), "Food"),
                                                          business_in_category(x.get("categories", ""), "Restaurants"),
                                                          business_in_category(x.get("categories", ""), "Coffee"),
                                                          business_in_category(x.get("categories", ""), "Nightlife"),
                                                          business_in_category(x.get("categories", ""), "Fast Food"),
                                                          business_in_category(x.get("categories", ""), "Bars"),


                                                          attribute_val(x.get("attributes"), "BusinessAcceptsCreditCards"),
                                                          attribute_val(x.get("attributes"), "RestaurantsDelivery"),
                                                          attribute_val(x.get("attributes"), "GoodForKids"),
                                                          attribute_val(x.get("attributes"), "RestaurantsReservations"))))


# creating dictionaries
user_file_dict = extract_user_features.collectAsMap()
business_file_dict = extract_business_features.collectAsMap()




def pairs_to_dict(pairs):
    result = {}
    for b, r in pairs:
        result[b] = r
    return result


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


user_threshold = 80 # 80 for task 2_2
business_threshold = 80 # 80 for task 2_2

def features_xgboost(user, business):
    u_avg_raw = user_avg_dict.get(user, all_avg)
    u_count = user_totals_dict.get(user, (0, 0))[1]

    b_avg_raw = business_avg_dict.get(business, all_avg)
    b_count = business_totals_dict.get(business, (0, 0))[1]

    u_smoothed = ((u_avg_raw * u_count) + (user_threshold * all_avg)) / (u_count + user_threshold)
    b_smoothed = ((b_avg_raw * b_count) + (business_threshold * all_avg)) / (b_count + business_threshold)

    # u_bias = u_smoothed - all_avg
    # b_bias = b_smoothed - all_avg

    diff_bias = u_smoothed - b_smoothed
    total_revs = u_count + b_count + 1
    user_ratio = u_count / total_revs
    business_ratio = b_count / total_revs



    user_features = user_file_dict.get(user, (0, 0, 0, 0, 0, all_avg))
    business_features = business_file_dict.get(business, (all_avg, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                          0, 0, 0, 0))

    # log_u_count = math.log1p(u_count)
    # log_b_count = math.log1p(b_count)
    # abs_diff = abs(u_smoothed - b_smoothed)
    # # smooth_diff = u_smoothed - b_smoothed
    # count_product = u_count * b_count
    #
    # # count_diff = u_count - b_count
    # user_dev = user_features[5] - u_smoothed
    # business_dev = business_features[0] - b_smoothed

    return [
        u_smoothed,
        b_smoothed,
        user_ratio,
        business_ratio,
        diff_bias,

        user_features[5],
        business_features[0],
        user_features[0],
        business_features[1],
        user_features[4],
        user_features[1],
        user_features[2],
        user_features[3],

        business_features[2],
        business_features[3],
        business_features[4],
        business_features[5],
        business_features[6],
        business_features[7],
        business_features[8],
        business_features[9],
        business_features[10],
        business_features[11],

        business_features[12],
        business_features[13]

        # business_features[14],
        # business_features[15],
        # business_features[16],
        # business_features[17],
        # business_features[18],
        # business_features[19],
        # business_features[20],
        # business_features[21],
        #
        #
        # business_features[22],
        # business_features[23],
        # business_features[24]
    ]


# predicting train file
def cf_prediction(user, business):
    # cold_case = {}

    # cold start issue we need to resolve
    # if both are missing return all avg
    if user not in user_dict and business not in business_dict:
        return all_avg
    # if user is missing return business avg
    if user not in user_dict:
        return business_avg_dict.get(business, all_avg)

    # if business is missing return user avg
    if business not in business_dict:
        return user_avg_dict.get(user, all_avg)

    get_user = user_dict[user]
    # get_business = business_dict.get[business]


    rated_items = list(get_user.items())

    # initializing
    numerator = 0.0
    denominator = 0.0


    for business_rating_pair in rated_items:
        business_part = business_rating_pair[0]
        rating_part = business_rating_pair[1]

        if business_part != business:
            similarity = pearson_similarity(business, business_part)
            if similarity > 0:
                numerator += similarity * rating_part
                denominator += abs(similarity)

    # if denominator == 0.0:
    #     preds = (user_avg_dict.get(user, all_avg) + business_avg_dict.get(business, all_avg)) / 2.0
    # else:
    #     preds = numerator / denominator

    if denominator == 0.0:
        preds = (user_avg_dict.get(user, all_avg) + business_avg_dict.get(business, all_avg)) / 2.0
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

# fit to the model to run predictions
# want to add one parameter to fine-tune model
xgb_regressor = XGBRegressor(
    max_depth=4,
    n_estimators=1000, # before 1000
    learning_rate=0.01, # before 0.01
    subsample=0.75, # before 0.7
    colsample_bytree=0.75, # before 0.7
    reg_lambda=5.0, # before 3.0
    nthread=1
    # n_jobs=1

)

xgb_regressor.fit(x_train, y_train)
testing = xgb_regressor.predict(x_test)


# adjusting for hybrid rec system
# alpha = 0.12

results = []
for i in range(len(test_prediction)):
    user = test_prediction[i][0]
    business = test_prediction[i][1]

    # prediction_num = testing[i]
    # #
    # if prediction_num < 1.0:
    #     prediction_num = 1.0
    # if prediction_num > 5.0:
    #     prediction_num = 5.0


    model = testing[i]
    cf_preds = cf_prediction(user, business)
    #
    # u_count = user_totals_dict.get(user, (0, 0))[1]
    # b_count = business_totals_dict.get(business, (0, 0))[1]
    #
    # if u_count < 5 or b_count < 5:
    #     prediction_num = model
    # else:
    #     alpha = 0.08
    #     prediction_num = alpha * cf_preds + (1 - alpha) * model

    if user not in user_dict:
        prediction_num = model
    elif business not in business_dict:
        prediction_num = model
    else:

        u_count = user_totals_dict.get(user, (0, 0))[1]
        b_count = business_totals_dict.get(business, (0, 0))[1]

        if u_count < 3 or b_count < 3:
            curr_alpha = 0.0
        elif u_count >= 20 and b_count >= 20:
            curr_alpha = 0.14
        elif u_count >= 10 and b_count >= 10:
            curr_alpha = 0.12
        else:
            curr_alpha = 0.06

        prediction_num = (curr_alpha * cf_preds) + ((1 - curr_alpha) * model)
    #     # prediction_num = (alpha * cf_preds) + ((1 - alpha) * model)
    #
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

output_file_write = open(output_file_name, "w")
output_file_write.write(final_all)
output_file_write.close()

end_time = time.time()
custom_exe_time = end_time - start_time
print("Duration:", custom_exe_time)