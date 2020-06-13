from matplotlib import pyplot as plt
import pickle

with open("running_time_pickles/times_cluster_centroids.pkl", 'rb') as f:
    times_sub_cluster = pickle.load(f)

with open("running_time_pickles/times_tomek_links.pkl", 'rb') as f2:
    times_sub_tomek_links = pickle.load(f2)

with open("running_time_pickles/times_smote_tomek.pkl", 'rb') as f3:
    times_sub_smote_tomek = pickle.load(f3)

with open("running_time_pickles/times_smote.pkl", 'rb') as f4:
    times_smote = pickle.load(f4)

with open("running_time_pickles/times_random.pkl", 'rb') as f5:
    times_random_dict = pickle.load(f5)
    times_random_over = times_random_dict['time_over']
    times_random_under = times_random_dict['time_under']
    times_random_over_under = times_random_dict['time_over_under']

with open("running_time_pickles/times_ADASYN.pkl", 'rb') as f6:
    times_adasyn = pickle.load(f6)

subset_list = [30000, 50000, 75000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1500000, 2000000]
times_random_over_zipped = zip(subset_list, times_random_over)
times_random_under_zipped = zip(subset_list, times_random_under)
times_random_over_under_zipped = zip(subset_list, times_random_over_under)

plt.figure()
plt.plot(*zip(*times_sub_cluster), '-o', label='Running time of Cluster centroids')
plt.plot(*zip(*times_sub_tomek_links), '-o', label='Running time of TomekLinks')
plt.plot(*zip(*times_sub_smote_tomek), '-o', label='Running time of SMOTE Tomek')
plt.plot(*zip(*times_smote), '-o', label='Running time of SMOTE')
plt.plot(*zip(*times_adasyn), '-o', label='Running time of ADASYN')
plt.plot(*zip(*times_random_over_zipped), '-o', label='Running time of Random Oversampling')
plt.plot(*zip(*times_random_under_zipped), '-o', label='Running time of Random Undersampling')
plt.plot(*zip(*times_random_over_under_zipped), '-o', label='Running time of Random Over and Undersampling')

plt.title('Running time of sampling method')
plt.ylabel("Running time in minutes")
plt.xlabel("Size of the subset in number of rows")
plt.xscale('log')
plt.legend()
plt.show()


