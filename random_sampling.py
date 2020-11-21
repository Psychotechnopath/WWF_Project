from general_functions import set_path_base, xg_boost
import pickle
import imblearn as imb
from sklearn.model_selection import train_test_split
import time
from multiprocessing import Process

path = set_path_base("Yme")
with open("{}/subset_x.pkl".format(path), "rb") as x:  # Import data
    X = pickle.load(x)
with open("{}/subset_y.pkl".format(path), "rb") as y:  # Import data
    y = pickle.load(y)

#Create a train-test split where the ratio of target class is maintained
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=47, stratify=y)

#Sampler_choice variable is used to determine which random sampling strategy is used.
#1 = random undersampling, 2 = random oversampling, 3 = random under and oversampling combined.
sampler_choice = 2
#test
#Sampling_strategy variable is used to set the sampling ratio.
sampling_strategy = 1
if sampler_choice == 1:
    #Initialize a random over sampler with ratio that will be tested
    under = imb.under_sampling.RandomUnderSampler(sampling_strategy=sampling_strategy)
    #Initialize a pipeline (One can add extra steps here if required)
    steps = [('u', under)]
    pipeline = imb.pipeline.Pipeline(steps)
    #Resample data
    x_train, y_train = pipeline.fit_resample(x_train, y_train)
    #Train an xg_boost model with resampled data
    xg_boost(x_train, y_train, x_test, y_test, 'Random Undersampling')
elif sampler_choice == 2:
    #Initialize a random under sampler with ratio that will be tested
    over = imb.over_sampling.RandomOverSampler(sampling_strategy=sampling_strategy)
    #Initialize a pipeline (One can add extra steps here if required)
    steps = [('o', over)]
    pipeline = imb.pipeline.Pipeline(steps)
    #Resample data
    x_train, y_train = pipeline.fit_resample(x_train, y_train)
    #Train an xg_boost model with resampled data
    xg_boost(x_train, y_train, x_test, y_test, 'Random Oversampling')
elif sampler_choice == 3:
    #Initialize a random over sampler, then a random undersampler with ratios that will be tested
    over = imb.over_sampling.RandomOverSampler(sampling_strategy=0.10)
    under = imb.under_sampling.RandomUnderSampler(sampling_strategy=sampling_strategy)
    #Initialize a pipeline (One can add extra steps here if required)
    steps = [('o', over), ('u', under)]
    pipeline = imb.pipeline.Pipeline(steps)
    #Resample data
    x_train, y_train = pipeline.fit_resample(x_train, y_train)
    #Train an xg_boost model with resampled data
    xg_boost(x_train, y_train, x_test, y_test, 'Random Over+Undersampling')


# The code below was used to calculate the running times.
# Since some running times were very long, we let the code time-out after 10 hours.
# # It is less relevant for WWF, hence it is commented out.

#To generate the running-time plots:
# subset_list = [30000, 50000, 75000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1500000, 2000000]
# times_subsetsize_dict = {'time_under': [],
#                          'time_over':[],
#                          'time_over_under': []}
#
# def do_actions():
#     for i in subset_list:
#         x_rest, x_sub, y_rest, y_sub = train_test_split(X, y, test_size=i/len(X), stratify=y, random_state=47)
#
#         #Random Undersampling
#         start_under = time.time()
#         print("Random_undersampling", i)
#         under = imb.under_sampling.RandomUnderSampler(sampling_strategy=0.042)
#         steps_under = [('u', under)]
#         pipeline_under = imb.pipeline.Pipeline(steps_under)
#         x_res_under, y_res_under = pipeline_under.fit_resample(x_sub, y_sub)
#         print("Resample finished")
#         stop_under = time.time()
#         time_under_sub = (stop_under-start_under)/60
#         times_subsetsize_dict['time_under'].append(time_under_sub)
#
#         #Random Oversampling
#         start_over = time.time()
#         print("Random_oversampling ", i)
#         over = imb.over_sampling.RandomOverSampler(sampling_strategy=0.042)
#         steps_over = [('o', over)]
#         pipeline_over = imb.pipeline.Pipeline(steps_over)
#         x_res_over, y_res_over = pipeline_over.fit_resample(x_sub, y_sub)
#         print("Resample finished")
#         stop_over = time.time()
#         time_over_sub = (stop_over-start_over)/60
#         times_subsetsize_dict['time_over'].append(time_over_sub)
#
#         #Random Over/Undersampling
#         start_over_under = time.time()
#         print("Random oversampling with random undersampling", i)
#         over_duplex = imb.over_sampling.RandomOverSampler(sampling_strategy=0.02)
#         under_duplex = imb.under_sampling.RandomUnderSampler(sampling_strategy=0.042)
#         steps_over_under = [('o', over_duplex), ('u', under_duplex)]
#         pipeline_over_under = imb.pipeline.Pipeline(steps_over_under)
#         x_res_over, y_res_over = pipeline_over_under.fit_resample(x_sub, y_sub)
#         print("Resample finished")
#         stop_over_under = time.time()
#         time_over_under_sub = (stop_over_under-start_over_under)/60
#         times_subsetsize_dict['time_over_under'].append(time_over_under_sub)
#
#
#         with open("running_time_pickles/times_random.pkl", 'wb') as f:
#             pickle.dump(times_subsetsize_dict, f)
#
#
# if __name__ == '__main__':
#     # We create a Process
#     action_process = Process(target=do_actions)
#     # We start the process and we block for 10 hours
#     action_process.start()
#     action_process.join(timeout=36000)
#     # We terminate the process.
#     action_process.terminate()
#     print("Hey there! I timed out! You can do things after me!")