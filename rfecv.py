import numpy as np
import pandas as pd
import argparse

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV

import os

parser = argparse.ArgumentParser(description='Specify term type')
parser.add_argument('--term', type=str, help="specify the surface termination A or B", required=True)

parser.add_argument('--rfecv', action='store_true', help="Perform recursive feature elimination "
                                                        "using fivefold CV, hyperparameters are "
                                                        "tuned at each step.")
parser.add_argument('--random_search', action='store_true', help="hyperparameters tuning using "
                                                                "random search")
parser.add_argument('--grid_search', action='store_true', help="hyperparameters tuning using "
                                                              "grid search")
args = parser.parse_args()

random_state = np.random.seed(123)


# Check and make directory
if not os.path.exists('rfecv_results'):
    os.makedirs('rfecv_results')


def grid_search(rf, X_train, y_train):  # grid search for optimization of hyper parameters

    print("Determine hyper parameters for RF Regressor with grid search...")

    param_dist = {'n_estimators': [100, 110, 120, 130],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth': range(10, 20, 2),
                'min_samples_split': [2, 3, 4],
                  }
    cv_rf = sklearn.model_selection.GridSearchCV(
        estimator = rf,
        cv = 5,
        param_grid = param_dist,
        n_jobs = -1,
        verbose = True)

    cv_rf.fit(X_train,y_train)
    print('Best Parameters using grid search: \n',
        cv_rf.best_params_)

    hyper = cv_rf.best_params_
    return hyper


def random_search(rf,X_train,y_train):  # random search for opmization of hyper prameters

    print("Determine hyper parameters for RF Regressor with random search...")

    param_dist = {'n_estimators': [int(x) for x in np.linspace(10, 120, num = 12)],
                  'max_features': ['auto', 'sqrt', 'log2'] + list(range(1, X_train.shape[1])),
                  'max_depth': range(1, 21),
                  'min_samples_split': [2,3,4],
                  }
    cv_rf = sklearn.model_selection.RandomizedSearchCV(
        estimator = rf,
        cv = 5,
        param_distributions = param_dist,
        n_jobs = -1,
        verbose = True,
        n_iter = 2)

    # {'max_depth': 10, 'max_features': 6, 'min_samples_split': 2, 'n_estimators': 90}
    cv_rf.fit(X_train,y_train)
    hyper = cv_rf.best_params_
    print("Best Parameters using randomized search: \n", cv_rf.best_params_)
    print("The best score across all searched params:\n", cv_rf.best_score_)
    return hyper

# -- read data and choose terminations
data = pd.read_csv("./data/perovskite_wf_data_uncorr.csv",index_col=0)

term_type = args.term
data = data[data.Type == term_type]

X = data.iloc[:,3:-1]
feature_names = list(X.columns)
Y = data.iloc[:,-1]
print(X.shape)
print(Y.shape)

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
print(X_train.shape)
print(y_train.shape)

#=====================================#
# initialize RFR
rfr = RandomForestRegressor(random_state=random_state)

if args.random_search:

    random_search(rfr, X_train, y_train)

if args.grid_search:

    grid_search(rfr, X_train, y_train)

if args.rfecv:  # recursive elimination, each step consists of hypyerparameter tuning;
              # don't use GridsearchCV nested with RFECV, instead, enumerate the
              # combination of hyper-parameter, and use use RFECV to do the CV

    import itertools
    print("Performing Recursive feature elimination...")
    print("Starting from {} samples and {} features...".format(X_train.shape[0], X_train.shape[1]))

    param_grid = {'n_estimators': [100, 110, 120, 130],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'max_depth': range(10, 20, 2),
                  'min_samples_split': [2,3,4],
                    }

    param_names = sorted(param_grid)
    combinations = list(itertools.product(*(param_grid[name] for name in param_names)))
    print("Grid contains {} hyper-parameter combinations...".format(len(combinations)))

    result_recorder = []
    featr_support = []
    rfetr_num = []

    counter = 1

    for i in combinations:
        print("CV for {} set hyperparameters...".format(counter))
        tmp_hyper = dict(zip(param_names, i))
        rfr_hyper = RandomForestRegressor(**tmp_hyper,random_state = random_state)
        rfe_cv = RFECV(estimator=rfr_hyper, step=1, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=False)

        rfe_cv.fit(X_train,y_train)
        result_recorder.append(rfe_cv.grid_scores_)  # record mean of CV
        featr_support.append(rfe_cv.support_)  # record selected features of CV
        rfetr_num.append(rfe_cv.n_features_)

        counter +=1

    result_recorder = np.array(result_recorder) * -1
    all_rmse = np.min(result_recorder,axis=0)  # record smallest RMSE

    rfecv_res = pd.DataFrame(np.column_stack([np.arange(1,len(all_rmse)+1),all_rmse]),
                columns = ["Number of features", "Mean RMSE"])

    rfecv_res.to_csv("rfecv_results/rfecv_term_{}.csv".format(term_type))  # for plotting purpose

    _tmp_featr_sel = []
    rfetr_num = np.array(rfetr_num)-1
    for idx,val in enumerate(result_recorder):
        _tmp_featr_sel.append(val[rfetr_num[idx]])

    best_perform_idx = np.argmin(_tmp_featr_sel)
    best_param = dict(zip(param_names, combinations[best_perform_idx]))
    best_featr = np.array(feature_names)[featr_support[best_perform_idx]].tolist()

    # print("dumping the trained model...")
    import json
    with open("rfecv_results/model_param_term_{}.json".format(term_type), 'w') as f:
        json.dump(best_param, f)


    col = list(data.columns[:3]) + best_featr
    transformed_data = data[col]
    transformed_data["Work function"] = data["Work function"]
    transformed_data.to_csv("data/transformed_data_term_{}.csv".format(term_type))