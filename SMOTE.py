from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from WWF_Project.deia2_general import set_path_base, to_dataframe, xg_boost

#%%
# Comment for discussion to be pushed:
# We should use SMOTENC to denote which features are nominal/categorical,
# otherwise the algorithm is going to interpolate between months, where month = 6.4 does not make any sense
import pickle
path_yme = set_path_base("Joost")
with open("{}/subset_x.pkl".format(path_yme), "rb") as x:  # Import data
    X = pickle.load(x)
with open("{}/subset_y.pkl".format(path_yme), "rb") as y:  # Import data
    y = pickle.load(y)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=47, stratify=y)
#Initialize a SMOTE sampler with
over = SMOTE(sampling_strategy=0.111)
#under = RandomUnderSampler(sampling_strategy=0.33)

steps = [ ('o', over)] #('u', under)
pipeline = Pipeline(steps)

x_res, y_res = pipeline.fit_resample(x_train, y_train)
print('resample finished')

xgb = xg_boost(x_res, y_res, x_test, y_test, "SMOTE_10%")




