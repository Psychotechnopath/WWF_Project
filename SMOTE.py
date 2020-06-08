from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from deia2_general import set_path_base, to_dataframe, xg_boost
import time
import pickle
from multiprocessing import Process

path = set_path_base("Joost")
with open("{}/subset_x.pkl".format(path), "rb") as x:  # Import data
    X = pickle.load(x)
with open("{}/subset_y.pkl".format(path), "rb") as y:  # Import data
    y = pickle.load(y)

#%%
#These ratio values correspond to the percentages of oversampling that were tested: 4%, 10%, 25%, 35% and 50%.
ratio_list = [0.042, 0.111, 0.333, 0.538, 1]
percentage_list = [4, 10, 25, 35, 50]


for ratio, percentage in zip(ratio_list, percentage_list):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=47, stratify=y)
    #Initialize a SMOTE sampler with
    over = SMOTE(sampling_strategy=ratio)
    steps = [ ('o', over)]
    pipeline = Pipeline(steps)
    x_res, y_res = pipeline.fit_resample(x_train, y_train)
    print('resample finished')
    xgb = xg_boost(x_res, y_res, x_test, y_test, f"SMOTE_{percentage}")

#To generate the running-time plots:
subset_list = [30000, 50000, 75000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1500000, 2000000]
times_subsetsize_list = []

#%%
def do_actions():
    for i in subset_list:
        start = time.time()
        x_rest, x_sub, y_rest, y_sub = train_test_split(X, y, test_size=i/len(X), stratify=y, random_state=47)

        print("SMOTE")
        over = SMOTE(sampling_strategy=ratio)
        steps = [('o', over)]
        pipeline = Pipeline(steps)
        x_res, y_res = pipeline.fit_resample(x_sub, y_sub)
        print("Resample finished")
        stop = time.time()
        times_subsetsize_list.append((i, (stop-start)/60))
        with open("running_time_pickles/times_smote.pkl", 'wb') as f:
            pickle.dump(times_subsetsize_list, f)


if __name__ == '__main__':
    # We create a Process
    action_process = Process(target=do_actions)
    # We start the process and we block for 10 hours
    action_process.start()
    action_process.join(timeout=36000)
    # We terminate the process.
    action_process.terminate()
    print("Hey there! I timed out! You can do things after me!")


