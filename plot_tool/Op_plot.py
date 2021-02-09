import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

a = pd.read_csv("A_Op_1d_pdp.csv",index_col=0)
b = pd.read_csv("B_Op_1d_pdp.csv",index_col=0)

#sns.scatterplot(a.index.values, a['0'].values, color="#EFC381", alpha = 0.2, s=80, linewidth=0.2, edgecolor="none")

plt.plot(a.index.values, a['0'].values, color="#EFC381", linewidth=4)
plt.plot(b.index.values, b['0'].values, color="#90A7CD", linewidth=4)
#sns.scatterplot(b.index.values, b['0'].values, color="#90A7CD", alpha = 0.2, s=80, linewidth=0.2, edgecolor="none")

plt.xlim(-7, 0)
plt.ylim(3, 6)
sns.despine(offset=10, trim=True)

fig = plt.gcf()
fig.set_size_inches(6, 6)

plt.savefig("./O2p_1d_pdp.png",dpi=300)
