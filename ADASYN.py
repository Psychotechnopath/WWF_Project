from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from general_functions import set_path_base, xg_boost
import pickle
import time
from multiprocessing import Process

#This code was used to load in the subset of data that we created (in make_subset.py)
#Replace with own mechanism to load in data
path = set_path_base("Yme")
with open("{}/subset_x.pkl".format(path), "rb") as x:
    X = pickle.load(x)
with open("{}/subset_y.pkl".format(path), "rb") as y:
    y = pickle.load(y)

# These ratio values correspond to the percentages of oversampling that were tested: 4%, 10%, 25%, 35% and 50%.
ratio_list = [0.042, 0.111, 0.333, 0.538, 1]
percentage_list = [4, 10, 25, 35, 50]

#This loop executes the oversampling strategy (In this case ADASYN) for all the ratio's that were tested.
for ratio, percentage in zip(ratio_list, percentage_list):
    #Create a train-test split where the ratio of target class is maintained
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=47, stratify=y)
    #Initialize a ADASYN sampler with ratio that will be tested
    over = ADASYN(sampling_strategy=ratio)
    #Initialize a pipeline (One can add extra steps here if required)
    steps = [ ('o', over)]
    pipeline = Pipeline(steps)
    #Resample data
    x_res, y_res = pipeline.fit_resample(x_train, y_train)
    print('resample finished')
    #Train an xg_boost model with resampled data
    xgb = xg_boost(x_res, y_res, x_test, y_test, f"ADASYN_{percentage}")


# The code below was used to calculate the running times.
# Since some running times were very long, we let the code time-out after 10 hours.
# It is less relevant for WWF, hence it is commented out.

#List of sub-sample sizes that were evaluated to calculate running times.
# subset_list = [30000, 50000, 75000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1500000, 2000000]
# times_subsetsize_list = []

# def calculate_running_times():
#     for i in subset_list:
#         start = time.time()
#         x_rest, x_sub, y_rest, y_sub = train_test_split(X, y, test_size=i/len(X), stratify=y, random_state=47)
#         print("ADASYN", i)
#         over = ADASYN(sampling_strategy=0.042)
#         steps = [('o', over)]
#         pipeline = Pipeline(steps)
#         x_res, y_res = pipeline.fit_resample(x_sub, y_sub)
#         print("Resample finished")
#         stop = time.time()
#         times_subsetsize_list.append((i, (stop-start)/60))
#         with open("running_time_pickles/times_ADASYN.pkl", 'wb') as f:
#             pickle.dump(times_subsetsize_list, f)


# if __name__ == '__main__':
#     # We create a Process
#     action_process = Process(target=calculate_running_times)
#     # We start the process and we block for 10 hours
#     action_process.start()
#     action_process.join(timeout=36000)
#     # We terminate the process.
#     action_process.terminate()
#     print("Hey there! I timed out! You can do things after me!")




