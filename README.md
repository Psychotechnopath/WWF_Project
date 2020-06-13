# WWF_Project
This is the GitHub repository for the WWF project. The code presented here was used to investigate the effectiveness of several sampling strategies. The README you are reading now attempts to give a comprehensive overview of the files in the repository, and the functions in those files. The dependencies that are required to run the code in this project can be found in the **requirements.txt**

The file **general_functions.py** was created for functions/code that was often (re) used. It contains three functions:
* set_path_base() was used to load in data from our local machines
* to_dataframe() was used to convert any pickle provided by WWF into a dataframe
* xg_boost() was used to train and evaluate the XGBoost model that was provided to us. This function is executed in all the resampling files.

The following files:
* **ADASYN.py**
* **cluster_centroids.py**
* **random_sampling.py**
* **SMOTE.py**
* **SMOTETomek.py**
* **TomekLinks.py**

contain the evaluated resampling strategies. 
The file names correspond to the evaluated sampling methods that were tested. 
All files have the same structure. 
First, the subset of data that we created is loaded in. 
Then we loop over all the different target class ratios that were evaluated (Except in TomekLinks, which uses a different strategy). 
For every target class ratio, we create a train-test-split, initialize a pipeline, and resample the data.
We then train an XGBoost model with the resampled data.
At the bottom of all files there is code that was used to evaluate the running times of different re-sampling strategies. It is not too relevant for WWF, hence we decided to comment it out. It is included for completeness. 


The file **make_subset.py** creates the subset of data that we used to evaluate all the sampling methods on. It loops over all the pickles, and takes a sub-sample of data from each pickle. As WWF will probably run the resampling methods on the entire dataset, this file is not relevant for WWF but for completeness we still provide it.

The file **run_time_plotter.py** was used to generate the plots that illustrate running times. It is not relevant for WWF but we include it for completeness.

The directory **running_time_pickles** contain the pickles with the data that was used to generate the running time plots. It is not relevant for WWF, but it is included for completeness.


Dependencies:
numpy==1.18.1
pandas==1.0.1
scikit-learn==0.22.1
xgboost==1.0.2
