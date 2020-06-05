from multiprocessing import Process
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
import imblearn.under_sampling
import time
import pickle
from deia2_general import set_path_base, to_dataframe, xg_boost

#%%
path_yme = set_path_base("Ellen")

with open("{}/subset_x.pkl".format(path_yme), "rb") as x:  # Import data
    X = pickle.load(x)
with open("{}/subset_y.pkl".format(path_yme), "rb") as y:  # Import data
    y = pickle.load(y)



subset_list = [30000, 50000, 75000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1500000, 2000000]
times_subsetsize_list = []

def do_actions():
    for i in subset_list:
        start = time.time()
        print("SMOTETomek")
        x_res, x_sub, y_res, y_sub = train_test_split(X, y, test_size=i/len(X), stratify=y, random_state=47)
        over = imblearn.combine.SMOTETomek(sampling_strategy=0.042)
        steps = [('o', over)]
        pipeline = Pipeline(steps)
        #[:i] stands for how much rows we will take in our subsets
        x_train, x_test, y_train, y_test = train_test_split(x_sub, y_sub, test_size=0.25, random_state=47, stratify=y_sub)

        x_train_res, y_train_res = pipeline.fit_resample(x_train, y_train)
        print("Resample finished")
        xg_boost(x_train_res, y_train_res, x_test, y_test, f"smote_tomek{i}")
        stop = time.time()
        times_subsetsize_list.append((i, (stop-start)/60))
        with open("times_smote_tomek.pkl", 'wb') as f:
            pickle.dump(times_subsetsize_list, f)




if __name__ == '__main__':
    # We create a Process
    action_process = Process(target=do_actions)
    # We start the process and we block for 10 hours
    action_process.start()
    action_process.join(timeout=60)
    # We terminate the process.
    action_process.terminate()
    print("Hey there! I timed out! You can do things after me!")







