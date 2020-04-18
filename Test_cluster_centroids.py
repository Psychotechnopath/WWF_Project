from collections import Counter
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=50000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1,
                           weights=[0.01, 0.99],
                           class_sep=0.8, random_state=0)
print(sorted(Counter(y).items()))

from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=0, sampling_strategy={0:400, 1:9600})
X_resampled, y_resampled = cc.fit_resample(X, y)
print(sorted(Counter(y_resampled).items()))
