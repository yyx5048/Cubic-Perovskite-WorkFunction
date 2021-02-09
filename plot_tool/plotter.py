import numpy as np
import pandas as pd
import argparse

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from os import path
import glob

parser = argparse.ArgumentParser(description='Plot type: --rfecv for recursive feature elimination, '
                                             '--pdp for 1d pdp plots.')
parser.add_argument('--rfecv', action='store_true', help="plot rfecv results")
parser.add_argument('--pdp', action='store_true', help="plot 1d pdp results")
args = parser.parse_args()

def rfecv_plot(rfecv_term_A, rfecv_term_B):

    rfe_a_dat = pd.read_csv(rfecv_term_A, index_col=0)
    rfe_b_dat = pd.read_csv(rfecv_term_B, index_col=0)

    fig, ax = plt.subplots(figsize=(8, 6))

    num_fea_a = rfe_a_dat["Number of features"]
    rmse_term_a = rfe_a_dat["Mean RMSE"]

    num_fea_b = rfe_b_dat["Number of features"]
    rmse_term_b = rfe_b_dat["Mean RMSE"]

    ax.scatter(num_fea_a, rmse_term_a, marker="o", c="#DC0000",
               s=60, edgecolors='white', linewidths=1.5,
               label="RMSE for A-termination")

    ax.plot(num_fea_a, rmse_term_a, marker="o", c="#DC0000", zorder=-1)

    ax.scatter(num_fea_b, rmse_term_b, marker="o", c="#00A087",
               s=60, edgecolors='white', linewidths=1.5,
               label="RMSE for B-termination")

    ax.plot(num_fea_b, rmse_term_b, marker="o", c="#00A087", zorder=-1)

    a_min_idx = np.argmin(rfe_a_dat["Mean RMSE"])
    a_min = rfe_a_dat["Mean RMSE"][a_min_idx]
    b_min_idx = np.argmin(rfe_b_dat["Mean RMSE"])
    b_min = rfe_a_dat["Mean RMSE"][b_min_idx]

    ax.arrow(a_min_idx+1, a_min + 0.1, 0, -0.07, color="#DC0000", zorder=-1)
    ax.arrow(b_min_idx+1, b_min + 0.1, 0, -0.07, color="#00A087", zorder=-1)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_title('Recursive feature elimination for AO and BO2 terminations')
    ax.set_xlabel('Number of features')
    ax.set_ylabel('RMSE of CV-work functions (eV)')

    plt.legend()
    sns.despine(offset=10, trim=False)

    plt.ylim(0.4, 1.4)
    plt.xlim(0, 24)
    plt.savefig("./rfe_plot.png", dpi=300)
    return fig


# 1d pdp plot
def pdp_1d_plotter(fname):
    term_type = fname.split("/")[-1][0]
    if term_type == "A":
        cl = "#EFC381"
    elif term_type == "B":
        cl = "#90A7CD"

    df = pd.read_csv(fname)
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(df.values[:, 0], df.values[:, 1], alpha=1.0, color=cl, facecolor="white",
               linewidth=2.5, s=100)
    sns.despine(offset=10, trim=True)

    plt.savefig(fname.split("/")[-1].split(".")[0] + ".png", dpi=300)
    return fig


if args.rfecv:
    rfecv_term_A = "../rfecv_results/rfecv_term_A.csv"
    rfecv_term_B = "../rfecv_results/rfecv_term_B.csv"

    if path.exists(rfecv_term_A) & path.exists(rfecv_term_B):
        rfecv_plt = rfecv_plot(rfecv_term_A, rfecv_term_B)
    else:
        print("Please run rfecv.py for feature elimination first...")

if args.pdp:
    res_dir = "../results/"
    pdp_list_1d = glob.glob(res_dir + "*1d_pdp.csv")

    if len(pdp_list_1d) > 0:
        for f in pdp_list_1d:
            pdp_1d_plotter(f)
    else:
        print("Please run model_analysis.py first...")
