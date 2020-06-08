from matplotlib import pyplot as plt
import pickle

# with open("times_sub_cluster.pkl", 'rb') as f:
#     times_sub_cluster = pickle.load(f)
#
# with open("running_time_pickles/times_sub_tomek_links.pkl", 'rb') as f2:
#     times_sub_tomek_links = pickle.load(f2)

with open("running_time_pickles/times_smote_tomek.pkl", 'rb') as f3:
    times_sub_smote_tomek = pickle.load(f3)



plt.figure()
#plt.plot(*zip(*times_sub_cluster), '-o', label='Running time of Cluster_centroids')
#plt.plot(*zip(*times_sub_tomek_links), '-o', label='Running time of TomekLinks')
plt.plot(*zip(*times_sub_smote_tomek), '-o', label='Running time of SMOTE Tomek')
plt.plot(*zip(*times_random_sampling), '-o', label='Running time of SMOTE Tomek')
plt.title('Running times of different cluster based sampling methods')
plt.ylabel("Running time")
plt.xlabel("Size of the subset")
# plt.xscale('log')
plt.legend()
plt.show()


