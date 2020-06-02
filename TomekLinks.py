from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
import imblearn.combine
import imblearn.under_sampling
import time
import pickle
from deia2_general import set_path_base, to_dataframe, xg_boost
from multiprocessing import Process

#%%
path_yme = set_path_base("Ellen")
with open("{}/subset_x.pkl".format(path_yme), "rb") as x:  # Import data
    X = pickle.load(x)
with open("{}/subset_y.pkl".format(path_yme), "rb") as y:  # Import data
    y = pickle.load(y)

subset_list = [1000, 5000, 10000, 15000, 20000, 25000, 50000, 75000, 100000, 200000, 300000, 4000000, 5000000, 600000, 700000, 800000, 900000, 1000000, 2000000, 3000000]
times_subsetsize_list = []

def do_actions():
    for i in subset_list:
        start = time.time()

        # Third pipeline Tomek links
        print("Tomek")
        over = imblearn.under_sampling.TomekLinks(sampling_strategy='majority')
        steps = [('o', over)]
        pipeline = Pipeline(steps)
        x_train, x_test, y_train, y_test = train_test_split(X[:i], y[:i], test_size=0.25, random_state=47)
        x_train_res, y_train_res = pipeline.fit_resample(x_train, y_train)

        print("Resample finished")
        xg_boost(x_train_res, y_train_res, x_test, y_test, f"tomek_links{i}")
        stop = time.time()
        times_subsetsize_list.append((i, (stop-start)/60))
        with open("times_sub_tomek_links.pkl", 'wb') as f:
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









