import numpy as np
import pandas as pd
import argparse
from joblib import dump, load
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from pdpbox import pdp

from matplotlib import pyplot as plt
import seaborn as sns

import os

parser = argparse.ArgumentParser(description='Specify term type')
parser.add_argument('--term', type=str, help="specify the surface termination A or B", required=True)

parser.add_argument('--pdp_1d', action='store_true', help="Perform 1d pdp plot for ")
parser.add_argument('--pdp_2d', action='store_true', help="Perform 2d pdp plot for ")
args = parser.parse_args()

if not os.path.exists('results'):
    os.makedirs('results')

random_state = np.random.seed(123)  # maybe range of random seed?


def evaluate(model, pred_train, pred_test, y_train, y_test):
    "Evaluate the performance of model"
    print('RMSE training and R^2:')
    train_RMSE = np.sqrt(mean_squared_error(y_train, pred_train))
    train_r2 = r2_score(y_train, pred_train)
    print(train_RMSE)
    print(train_r2)

    print('RMSE testing and R^2:')
    test_RMSE = np.sqrt(mean_squared_error(y_test, pred_test))
    test_r2 = r2_score(y_test, pred_test)
    print(test_RMSE)
    print(test_r2)

    fea_rank = np.argsort(model.feature_importances_)[::-1]
    fea_score = model.feature_importances_

    return fea_rank, fea_score, train_RMSE, train_r2, test_RMSE, test_r2


def importance_plot(term_type, ranked_fea, ranked_weight, num):
    if term_type == "A":
        cl = "#EFC381"
    elif term_type == "B":
        cl = "#90A7CD"

    imp_df = pd.DataFrame({"features": ranked_fea,
                           "importance": ranked_weight})

    imp_df.to_csv("./results/{}_importance.csv".format(term_type))

    ranked_fea = ranked_fea[:num]
    ranked_weight = ranked_weight[:num]

    fig, ax = plt.subplots(figsize=(4, 6))

    ax.set_title('Feature Importance of {}-termination'.format(term_type))
    ax.barh(ranked_fea, ranked_weight, color=cl, height=0.5, align='center', edgecolor='w')
    ax.set_xlabel('Relative Importance')
    plt.gca().invert_yaxis()
    # ax.yticks(ranked_fea), ranked_fea))

    fig.savefig("./results/{}_importance.png".format(term_type), dpi=300)
    return fig


def actual_vs_predict_plot(term_type, true_y_df, pred_y_df):
    if term_type == "A":
        cl = "#EFC381"
        ma = "^"
    elif term_type == "B":
        cl = "#90A7CD"
        ma = "o"

    true_y = true_y_df.mean(axis=1)
    pred_y = pred_y_df.mean(axis=1)

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.xlim(0.0, 11)
    plt.ylim(0.0, 11)
    sns.scatterplot(x=pred_y, y=true_y, color=cl, marker=ma, alpha=1.0)
    plt.plot(np.arange(0, 11), np.arange(0, 11), '--', color='black')

    sns.despine(offset=10, trim=True)
    ax.set_title('Actual vs predict plot for {}-termination'.format(term_type))
    ax.set_xlabel('Predicted work function (eV)')
    ax.set_ylabel('Calculated work function (eV)')

    fig.savefig("./results/{}_actual_vs_true.png".format(term_type), dpi=300)

    return fig


def pdp_1d_plot(term_type, pdp_df, fea_1d_name):
    if term_type == "A":
        cl = "#EFC381"
    elif term_type == "B":
        cl = "#90A7CD"
    pdp_df = pdp_df.mean(axis=1)
    pdp_df.to_csv("./results/{}_{}_1d_pdp.csv".format(term_type, fea_1d_name))

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(pdp_df.index, pdp_df.values, ax=ax, color=cl, s=80, linewidth=0.2, edgecolor="black")
    plt.xlim(-7, 0)
    plt.ylim(3, 6)
    sns.despine(offset=10, trim=True)
    fig.savefig("./results/{}_{}_1d_pdp.png".format(term_type, fea_1d_name), dpi=300)
    return fig

def pdp_2d_pdp(term_type, train_data, fea_2d_1, fea_2d_2, fea_nam):

    fea_1_min = min(train_data[fea_2d_1].values)
    fea_1_max = max(train_data[fea_2d_1].values)
    fea_2_min = min(train_data[fea_2d_2].values)
    fea_2_max = max(train_data[fea_2d_2].values)

    inter_rf = pdp.pdp_interact(
        model=rfr, dataset=train_data, model_features=fea_nam,
        cust_grid_points=[np.linspace(fea_1_min, fea_1_max, 10), np.linspace(fea_2_min, fea_2_max, 10)],
        features=[fea_2d_1, fea_2d_2])

    fig, axes = pdp.pdp_interact_plot(
        inter_rf, [fea_2d_1, fea_2d_2], x_quantile=False, plot_type='contour', plot_pdp=False)

    fig.savefig("./results/{}_{}-{}_2d_pdp.png".format(term_type, fea_2d_1, fea_2d_2), dpi=300)
    return fig

# --perform the random forest regression

term_type = args.term
with open("./rfecv_results/model_param_term_{}.json".format(term_type)) as f:
    param = json.load(f)
data = pd.read_csv("./data/transformed_data_term_{}.csv".format(term_type), index_col=0)
data.reset_index(inplace=True, drop=True)

X = data.iloc[:, 3:-1]
feature_names = list(X.columns)
Y = data.iloc[:, -1]
print(X.shape)
print(Y.shape)

if args.pdp_1d:
    print(X.columns)
    pdp_1d_fea = input("enter pdp 1d plot features..")  # 1d pdp analysis feature


if args.pdp_2d:
    print(X.columns)
    pdp_2d_fea_1 = input("enter 1st feature for pdp plot..")  # 2d pdp analysis feature
    pdp_2d_fea_2 = input("enter 2nd feature for pdp plot..")

if not os.path.exists('model_checkpoints'):
    os.makedirs('model_checkpoints')

rfr = RandomForestRegressor(**param, random_state=random_state)

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
rfr.fit(X_train, y_train)

dump(rfr, './model_checkpoints/perov_wf_model_term_{}.joblib'.format(term_type))

if args.pdp_2d:

    pdp_2dp = pdp_2d_pdp(term_type, X_train, pdp_2d_fea_1, pdp_2d_fea_2, feature_names)

# evaluate the model performance and record mean and std for the model, and evaluate the importance ranking.
# place holder for results
total_r2 = []
total_rmse = []
total_importance_score = np.zeros(len(feature_names))

total_pred_test = pd.DataFrame()  # use df index here to sort the average results
total_actual_test = pd.DataFrame()
pdp_1d_data = pd.DataFrame()

cycle = 40
for rnd_sd in range(cycle):
    random_state = np.random.seed(rnd_sd)  # generate data 40 times, give total 200 evaluations for fivefold CV
    # record mean and std for the model, do importance ranking plot
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    rfr_perform = RandomForestRegressor(**param, random_state=random_state)

    rfr_perform.fit(X_train, y_train)
    pred_train = rfr_perform.predict(X_train)
    pred_test = rfr_perform.predict(X_test)

    fea_rank, fea_weight, _train_rmse, _train_r2, _test_rmse, _test_r2 = evaluate(rfr_perform, pred_train, pred_test,
                                                                                  y_train, y_test)

    total_importance_score += fea_weight
    total_r2.append(_test_r2)
    total_rmse.append(_test_rmse)

    total_actual_test = total_actual_test.merge(y_test.to_frame(), how='outer', left_index=True, right_index=True)

    pred_test = pd.DataFrame(pred_test, index=y_test.index, columns=["Work function"])
    total_pred_test = total_pred_test.merge(pred_test, how='outer', left_index=True, right_index=True)

    # perform 1d pdp analysis

    if args.pdp_1d:
        print("perform 1d pdp analysis of cycle {}...".format(rnd_sd))
        pdp_1d_analysis = pdp.pdp_isolate(
            model=rfr_perform, dataset=X_train, model_features=feature_names, feature=pdp_1d_fea,
            cust_grid_points=np.unique(X_train[pdp_1d_fea].values)
        )
        pdp_1d_tmp = pd.DataFrame(pdp_1d_analysis.pdp,index=pdp_1d_analysis.feature_grids,columns=["1d pdp"])
        pdp_1d_data = pdp_1d_data.merge(pdp_1d_tmp, how='outer', left_index=True, right_index=True)


total_importance_score = total_importance_score / cycle

print("averaged R2 and rmse are {} and {}.".format(np.mean(total_r2), np.mean(total_rmse)))
print("Standard deviation of R2 and rmse are {} and {}.".format(np.std(total_r2), np.std(total_rmse)))

total_fea_rank = np.argsort(total_importance_score)[::-1]
ranked_featr = np.array(feature_names)[total_fea_rank]
ranked_featr_imp = total_importance_score[total_fea_rank]

# actual vs predict plot
act_vs_pred_fig = actual_vs_predict_plot(term_type, total_actual_test, total_pred_test)

# importance ranking plot
imp_fig = importance_plot(term_type, ranked_featr, ranked_featr_imp, 6)

if args.pdp_1d:
    pdp_1dp = pdp_1d_plot(term_type, pdp_1d_data, pdp_1d_fea)