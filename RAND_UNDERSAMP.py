import deia2_general as d2g
import pickle
import imblearn as imb
from sklearn.model_selection import train_test_split

path = d2g.set_path_base('Tim') + 'subset_x.pkl'
with open(path, "rb") as f:
    df = pickle.load(f)
path = d2g.set_path_base('Tim') + 'subset_y.pkl'
with open(path, "rb") as f:
    y = pickle.load(f)
X = df.copy(deep=True)
y = y['future_deforestation']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)


sampler_choice = 1  # 1 = Under, 2 = Over, 3 = Both

if sampler_choice == 1:
    under = imb.under_sampling.RandomUnderSampler(sampling_strategy=1)
    steps = [('u', under)]
    pipeline = imb.pipeline.Pipeline(steps)
    X, y = pipeline.fit_resample(x_train, y_train)
    d2g.xg_boost(x_train, y_train, x_test, y_test, 'Random Undersampling')


if sampler_choice == 2:
    over = imb.over_sampling.RandomOverSampler(sampling_strategy=0.04166666666)  # 4% minority after resampling
    steps = [('o', over)]
    pipeline = imb.pipeline.Pipeline(steps)
    X, y = pipeline.fit_resample(x_train, y_train)
    d2g.xg_boost(x_train, y_train, x_test, y_test, 'Random Oversampling')

elif sampler_choice == 3:
    over = imb.over_sampling.RandomOverSampler(sampling_strategy=0.10)  # NOT UP-TO-DATE
    under = imb.under_sampling.RandomUnderSampler(sampling_strategy=0.04166666666)  # NOT UP-TO-DATE
    steps = [('o', over), ('u', under)]
    pipeline = imb.pipeline.Pipeline(steps)
    X, y = pipeline.fit_resample(x_train, y_train)
    d2g.xg_boost(x_train, y_train, x_test, y_test, 'Random Over+Undersampling')
