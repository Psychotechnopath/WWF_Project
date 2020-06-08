from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
import imblearn.combine
import imblearn.under_sampling
import time
import pickle
from deia2_general import set_path_base, to_dataframe, xg_boost
from multiprocessing import Process

path = set_path_base("Yme")

with open("{}/subset_x.pkl".format(path), "rb") as x:  # Import data
    X = pickle.load(x)
with open("{}/subset_y.pkl".format(path), "rb") as y:  # Import data
    y = pickle.load(y)


#To run the full model:
under = imblearn.under_sampling.TomekLinks(sampling_strategy='majority')
steps = [('o', under)]
pipeline = Pipeline(steps)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=47, stratify=y)
x_train_res, y_train_res = pipeline.fit_resample(x_train, y_train)
xg_boost(x_train_res, y_train_res, x_test, y_test, f"tomek_links{len(X)}")


#To generate the running-time plots:
subset_list = [30000, 50000, 75000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1500000, 2000000]
times_subsetsize_list = []

def do_actions():
    for i in subset_list:
        start = time.time()
        x_rest, x_sub, y_rest, y_sub = train_test_split(X, y, test_size=i/len(X), stratify=y, random_state=47)
        # Third pipeline Tomek links
        print("Tomek")
        over = imblearn.under_sampling.TomekLinks(sampling_strategy='majority')
        steps = [('o', over)]
        pipeline = Pipeline(steps)
        x_train_res, y_train_res = pipeline.fit_resample(x_sub, y_sub)
        print("Resample finished")
        stop = time.time()
        times_subsetsize_list.append((i, (stop-start)/60))
        with open("running_time_pickles/times_tomek_links.pkl", 'wb') as f:
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









