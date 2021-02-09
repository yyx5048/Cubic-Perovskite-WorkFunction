import numpy as np
import pandas as pd
import argparse

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Remove correlated feature:, '
                                             '--s [u,e], specify the strategies:'
                                             'u uses upper triangle method, e uses'
                                             'enumeration method, default is enumeration'
                                             'method.')

parser.add_argument('-m', type=str, help="u uses upper triangle method, e uses enumeration "
                                          "method, default is enumeration")

parser.add_argument('-p', action='store_true', help="plot correlation matrix")

args = parser.parse_args()

data = pd.read_csv("./data/perovskite_wf_data.csv", index_col=0)
featr = data.iloc[:, 3:-1]

print(featr.shape)


def upper_tri_method(df, thr):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > thr)]
    return to_drop


def enumerate_method_inner(df, thr):
    for f in df.columns:
        corr_matrix = df.corr().abs()
        fea_column = corr_matrix[f]
        tmp = fea_column[fea_column > thr]
        tmp_drop = list(tmp.index.values)
        if len(tmp_drop) > 1:
            tmp_drop.remove(f)
            #print(tmp_drop)
            df = df.drop(df[tmp_drop], axis=1)
            break
    return df


def enumerate_method_outer(df, thr):
    while True:
        ori_shape = df.shape
        df = enumerate_method_inner(df, thr)
        if df.shape == ori_shape:
            break
    return df.columns.values


def corr_plot(corr_mat):
    c = corrplot.Corrplot(corr_mat)
    c.plot(shrink=.9, rotation=45, method='circle', cmap='viridis', grid="grey")
    fig = plt.gcf()
    fig.set_size_inches(18, 10, forward=True)
    plt.savefig('./results/pearson_corr.png', dpi=300)
    return

# By default enumeration method is used based on Pearson coefficient


if args.s == "u":
    print("using upper triangle method..")
    uncorrelated_features = upper_tri_method(featr, 0.8)  # remove features if abs(Pearson) > 0.8
elif args.s == "e" or args.s == None:
    print("using enumeration method..")
    uncorrelated_features = enumerate_method_outer(featr, 0.8)
else:
    print("invalid strategy type.., chose from u or e, default is e [elimination method]")

col = list(data.columns[:3]) + list(uncorrelated_features)
col.append(data.columns[-1])
data = data[col]

data.to_csv("../data/perovskite_wf_data_uncorr.csv")
if args.p:
    from biokit.viz import corrplot
    corr_plot(featr.corr())
