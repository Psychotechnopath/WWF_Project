from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from WWF_Project.deia2_general import set_path_base, to_dataframe, xg_boost
import pandas as pd

# Comment for discussion to be pushed:
# We should use SMOTENC to denote which features are nominal/categorical,
# otherwise the algorithm is going to interpolate between months, where month = 6.4 does not make any sense
import pickle

#%%
path = set_path_base('Joost')

#%%
with open(f'{path}/subset_x.pkl','rb') as f:
    X = pickle.load(f)
