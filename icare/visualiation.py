import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

def plot_avg_sign(model, features=None):
    dfimp = model.average_feature_signs_
    if features is not None:
        dfimp = dfimp[dfimp['feature'].isin(features)]

    mean_fp = dfimp.groupby(dfimp['feature'])
    mean_fp = mean_fp.mean().sort_values(by='average sign', ascending=False)


    all_v = []
    all_f = []
    for f in dfimp['feature'].unique():
        v = dfimp[dfimp['feature']==f]['average sign'].mean()
        all_v.append(v)
        all_f.append(f)
    all_v = np.array(all_v)
    all_f = np.array(all_f)
    idxsort = np.argsort(all_v)
    all_v = all_v[idxsort]
    all_f = all_f[idxsort]
    vabs = 1.3
    norm = matplotlib.colors.Normalize(vmin=-vabs, vmax=vabs)
    custom_palette = {}
    for i in range(len(all_f)):
        custom_palette[all_f[i]] = matplotlib.cm.get_cmap('RdBu_r')(norm(all_v[i]))

    sns.set_style("whitegrid")


    sns.barplot(data=dfimp.iloc[:], y="feature", x="average sign", order=mean_fp.index, errwidth=0, edgecolor='black', linewidth=0.5, palette=custom_palette)
    plt.xlim(-1.1,1.1)
    plt.xlabel('Average sign')
    plt.ylabel('Feature')
    plt.grid(axis = 'y')