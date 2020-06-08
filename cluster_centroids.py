from deia2_general import to_dataframe, set_path_base, xg_boost
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import ClusterCentroids
from multiprocessing import Process
import time
import pickle


base_path = set_path_base("Yme")

with open(f'{base_path}subset_x.pkl', 'rb') as f:
    X = pickle.load(f)

with open(f'{base_path}subset_y.pkl', 'rb') as f2:
    y = pickle.load(f2)

subset_list = [30000, 50000, 75000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1500000, 2000000]
times_subsetsize_list = []

def do_actions():
    for i in subset_list:
        start = time.time()
        x_res, x_sub, y_res, y_sub = train_test_split(X, y, test_size=i/len(X), stratify=y, random_state=47)
        cc = ClusterCentroids(sampling_strategy=0.042)


        #[:i] stands for how much rows we will take in our subsets
        x_train, x_test, y_train, y_test = train_test_split(x_sub, y_sub, test_size=0.25, random_state=47, stratify=y_sub)
        x_train_res, y_train_res = cc.fit_sample(x_train, y_train)
        xg_boost(x_train_res, y_train_res, x_test, y_test, f"cluster_centroids{i}")
        stop = time.time()
        times_subsetsize_list.append((i, (stop-start)/60))

        with open("running_time_pickles/times_sub_cluster.pkl", 'wb') as f3:
            pickle.dump(times_subsetsize_list, f3)


if __name__ == '__main__':
    # We create a Process
    action_process = Process(target=do_actions)
    # We start the process and we block for 10 hours
    action_process.start()
    action_process.join(timeout=36000)
    # We terminate the process.
    action_process.terminate()
    print("Hey there! I timed out! You can do things after me!")
