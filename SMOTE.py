from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from deia2_general import set_path_base, to_dataframe, xg_boost

# Comment for discussion to be pushed:
# We should use SMOTENC to denote which features are nominal/categorical,
# otherwise the algorithm is going to interpolate between months, where month = 6.4 does not make any sense
import pickle

#LEAVE FOR YME
# with open("Dict_Model_object.pkl" , "rb") as f:
#     unpickled = pickle.load(f)
#     print(unpickled.keys())
#     print(unpickled['params'])
#
# with open("subset_x.pkl", "rb") as f2:
#     X = pickle.load(f2)
#
# with open ("subset_y.pkl", "rb") as f3:
#     y = pickle.load(f3)

path_yme = set_path_base("Yme")
df = to_dataframe("{}/TilePickle_25.pkl".format(path_yme))
df.dropna(inplace=True)


#Set correct predicted and predictor variables
y = df['future_deforestation']
df.drop(columns=['future_deforestation'], inplace=True)
X = df

#LEAVE FOR YME
# print(len(X))
# X.dropna(inplace=True)
# print(len(X))

#Initialize a SMOTE sampler with
over = SMOTE(sampling_strategy=0.1)
#under = RandomUnderSampler(sampling_strategy=0.33)

steps = [ ('o', over)] #('u', under)
pipeline = Pipeline(steps)

X, y = pipeline.fit_resample(X, y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=47)

xgb = xg_boost(x_train, y_train, x_test, y_test, "SMOTE_10%")


#LEAVE FOR YME
# xgb_joanne = unpickled['bst']
# x_test_dm = xgb.DMatrix(x_test)
#
# y_pred = xgb_joanne.predict(x_test)
# # conf_matrix = confusion_matrix(y_test, y_pred)
# # accuracy = accuracy_score(y_test, y_pred, normalize=True)
# # print('Accuracy Joanne xgb: {} '.format(accuracy_score(y_test, y_pred)))
# # sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
# # print('Sensitivity Joanne xgb: {}'.format(sensitivity))
# # specificity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
# # print('Specificity Joanne xgb: {}'.format(specificity))
# # print(f"{conf_matrix} JOANNNE")



