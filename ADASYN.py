# The intuition:
## 1 https://medium.com/coinmonks/smote-and-adasyn-handling-imbalanced-data-set-34f5223e167
## 2 https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html#smote-adasyn

# So, summarized:
## 1 ADASYN is SMOTE, but adds a bit of noise to each synthetically generated data point

## 2 ADASYN focuses on generating samples next to the original samples which are
# wrongly classified using a k-Nearest Neighbors classifier,
# while SMOTE will not make any distinction between easy and hard samples using the nearest neighbors rule.


import pandas as pd
import pickle
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.metrics import geometric_mean_score
import xgboost as xgb
import winsound

winsound.Beep(500, 1000)

# Import data
with open("D:/Downloads/DEiA/TilePickle_25.pkl", "rb") as f:
    tile = pickle.load(f)

    # Flatten matrix
    reshaped_tile = tile.reshape(tile.shape[0], (tile.shape[1] * tile.shape[2]))
    tile = 0

    # Transform to dataframe
    df = pd.DataFrame(reshaped_tile.T, columns=['AggIndex1', 'AggIndex2', 'AggIndex3',
                                                'AggIndex4', 'AggIndex5', 'EdgeDensity1',
                                                'EdgeDensity2', 'EdgeDensity3', 'EdgeDensity4',
                                                'PatchDensity1', 'PatchDensity2', 'PatchDensity3',
                                                'PatchDensity4', 'LandcoverPercentage1', 'LandcoverPercentage2',
                                                'LandcoverPercentage3', 'LandcoverPercentage4', 'ShannonDiversity',
                                                'current_deforestationDistance', 'current_degradationDistance',
                                                'future_deforestation', 'RawSarVisionClasses', 'SarvisionBasemap',
                                                'scaledPopDensity', 'scaledASTER', 'RoadsDistance',
                                                'UrbanicityDistance',
                                                'WaterwaysDistance', 'CoastlineDistance', 'MillDistance',
                                                'PalmOilConcession',
                                                'gradientASTER', 'LogRoadDistance', 'Vegetype', 'CurrentMonth',
                                                'y_center',
                                                'x_center', 'time', 'size'])
    reshaped_tile = 0
    # Just a small check
    print(df.shape)
    print(df['future_deforestation'].value_counts())

#
# with open("./partial_pickle_25.pkl", "rb") as f:
#     df = pd.DataFrame(pickle.load(f))
#     # Just a small check
#     print(df.shape)
#     print(df['future_deforestation'].value_counts())

# Memory issues are basically non-existent, so df can keep existing :)
y = df['future_deforestation']
X = df.drop(columns=['future_deforestation'])

# we take a random train test split of 25%
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=47)

print(y_train.mean()) # what percentage of training data has deforestation
print(y_test.mean()) # and what percentage of testing data has deforestation (should be rel. close)
print(y_test.value_counts())

# The meat of the file: naive train-test-split + xgboost fitting
# Input are the training data, either in original shape or resampled
# The testing data are identical for every pipeline
def train_xgb(x_resampled, y_resampled):
    xgb_model = xgb.XGBClassifier(objective='reg:squarederror', random_state=47)
    xgb_model.fit(x_resampled, y_resampled)
    #xgb_model.save_model('trained_model')
    y_pred = xgb_model.predict(x_test)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    print('The geometric mean is {}'.format(geometric_mean_score(
        y_test,
        y_pred)))



# # First pipeline: just all data unedited
# print("No resampling")
# train_xgb(x_train, y_train)
#
# # Second pipeline: only ADASYN oversampling to minority == 10% of data
# print("ADASYN 0.1 oversampling")
# over_2 = ADASYN(sampling_strategy=0.1)
# steps_2 = [ ('o', over_2)]
# pipeline_2 = Pipeline(steps_2)
# x_r_2, y_r_2 = pipeline_2.fit_resample(x_train, y_train)
# train_xgb(x_r_2, y_r_2)
#
# # Third pipeline: only SMOTE oversampling to minority == 10% of data
# print("SMOTE 0.1 oversampling")
# over_3 = SMOTE(sampling_strategy=0.1)
# steps_3 = [ ('o', over_3)]
# pipeline_3 = Pipeline(steps_3)
# x_r_3, y_r_3 = pipeline_3.fit_resample(x_train, y_train)
# train_xgb(x_r_3, y_r_3)
#
# # Third pipeline: only SMOTE oversampling to minority == 10% of data
# print("SMOTE 0.3 oversampling")
# over_3 = SMOTE(sampling_strategy=0.3)
# steps_3 = [ ('o', over_3)]
# pipeline_3 = Pipeline(steps_3)
# x_r_3, y_r_3 = pipeline_3.fit_resample(x_train, y_train)
# train_xgb(x_r_3, y_r_3)
#
# # Third pipeline: only SMOTE oversampling to minority == 10% of data
# print("SMOTE 0.5 oversampling")
# over_3 = SMOTE(sampling_strategy=0.5)
# steps_3 = [ ('o', over_3)]
# pipeline_3 = Pipeline(steps_3)
# x_r_3, y_r_3 = pipeline_3.fit_resample(x_train, y_train)
# train_xgb(x_r_3, y_r_3)
#
# # Fourth pipeline: only randomly undersampling to minority == 10% of data
# print("RUS 0.1")
# under_4 = RandomUnderSampler(sampling_strategy=0.10)
# steps_4 = [ ('u', under_4)]
# pipeline_4 = Pipeline(steps_4)
# x_r_4, y_r_4 = pipeline_4.fit_resample(x_train, y_train)
# train_xgb(x_r_4, y_r_4)
#
# # Fifth pipeline: both ADASYN oversampling and random undersampling
# # such that min class == 10% of data
# print("ADASYN 0.05 oversampling + RUS 0.1")
# over_5 = ADASYN(sampling_strategy=0.05)
# under_5 = RandomUnderSampler(sampling_strategy=0.1)
# steps_5 = [ ('o', over_5), ('u', under_5)]
# pipeline_5 = Pipeline(steps_5)
# x_r_5, y_r_5 = pipeline_5.fit_resample(x_train, y_train)
# train_xgb(x_r_5, y_r_5)
#
# # Sixth pipeline: both SMOTE oversampling and random undersampling
# # such that min class == 10% of data
# print("SMOTE 0.05 oversampling + RUS 0.1")
# over_6 = SMOTE(sampling_strategy=0.05)
# under_6 = RandomUnderSampler(sampling_strategy=0.1)
# steps_6 = [ ('o', over_6), ('u', under_6)]
# pipeline_6 = Pipeline(steps_6)
# x_r_6, y_r_6 = pipeline_6.fit_resample(x_train, y_train)
# train_xgb(x_r_6, y_r_6)

for j in [10]: #nearest neighbors
    for i in [0.5]:
        print("ADASYN sampling ratio {}".format(i))
        print("ADASYN knn {}".format(j))
        adasyn = ADASYN(sampling_strategy=i, n_neighbors=j)
        steps = [('u', adasyn)]
        pipeline = Pipeline(steps)
        x_r, y_r = pipeline.fit_resample(x_train, y_train)
        train_xgb(x_r, y_r)




# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import model_selection as ms
# from sklearn import datasets, metrics, tree
#
# from imblearn import over_sampling as os
# from imblearn import pipeline as pl
#
# RANDOM_STATE = 42
#
# scorer = metrics.make_scorer(metrics.cohen_kappa_score)
#
# adasyn = os.ADASYN(random_state=RANDOM_STATE)
# xgb = xgb.XGBClassifier(random_state=RANDOM_STATE)
# pipeline = pl.make_pipeline(adasyn, xgb)
# winsound.Beep(500, 100)

#         print("ADASYN knn {}".format(j))
#         adasyn = ADASYN(sampling_strategy=i, n_neighbors=j)
#         steps = [('u', adasyn)]
#         pipeline = Pipeline(steps)
#         x_r, y_r = pipeline.fit_resample(x_train, y_train)
#


# param_range = range(1, 12)
# train_scores, test_scores = ms.validation_curve(
#     pipeline, X, y, param_name="adasyn__n_neighbors", param_range=param_range,
#     cv=3, scoring=scorer, n_jobs=1)
#
#
#
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# winsound.Beep(500, 300)
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# plt.plot(param_range, test_scores_mean, label='ADASYN')
# ax.fill_between(param_range, test_scores_mean + test_scores_std,
#                 test_scores_mean - test_scores_std, alpha=0.2)
# idx_max = np.argmax(test_scores_mean)
# plt.scatter(param_range[idx_max], test_scores_mean[idx_max],
#             label=r'Cohen Kappa: ${:.2f}\pm{:.2f}$'.format(
#                 test_scores_mean[idx_max], test_scores_std[idx_max]))
#
# plt.title("Validation Curve with ADASYN-XGB")
# plt.xlabel("n_neighbors")
# plt.ylabel("Cohen's kappa")
#
# # make nice plotting
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.get_xaxis().tick_bottom()
# ax.get_yaxis().tick_left()
# ax.spines['left'].set_position(('outward', 10))
# ax.spines['bottom'].set_position(('outward', 10))
# plt.xlim([1, 10])
# plt.ylim([0.4, 0.8])
#
# plt.legend(loc="best")
# plt.show()
# winsound.Beep(500, 3000)