import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt
import os
os.sys.path.append('.')

# def convert_pvalue_to_asterisks(pvalue):
#     if pvalue <= 0.0001:
#         return "****"
#     elif pvalue <= 0.001:
#         return "***"
#     elif pvalue <= 0.01:
#         return "**"
#     elif pvalue <= 0.05:
#         return "*"
#     return "ns"

print('start significance analysis')
result_EUNet = np.loadtxt('./outputs/results/EUNet/EUNet_EMG.txt')
result_MINDS = np.loadtxt('./outputs/results/MINDS/MINDS_EMG.txt')
result_MKCNN = np.loadtxt('./outputs/results/MKCNN/MKCNN_EMG.txt')
result_MSCNN = np.loadtxt('./outputs/results/MSCNN/MSCNN_EMG.txt')
result_XceptionTime = np.loadtxt('./outputs/results/XceptionTime/XceptionTime_EMG.txt')
avg_EUNet = np.mean(result_EUNet, axis = 1)
avg_MINDS = np.mean(result_MINDS, axis = 1)
avg_MKCNN = np.mean(result_MKCNN, axis = 1)
avg_MSCNN = np.mean(result_MSCNN, axis = 1)
avg_XceptionTime = np.mean(result_XceptionTime, axis = 1)

models = ['MSCNN', 'EUNet', 'MKCNN', 'XceptionTime', 'MINDS']
avg_res = np.hstack([avg_MSCNN, avg_EUNet, avg_MKCNN, avg_XceptionTime, avg_MINDS])
df_res = pd.DataFrame({'Model':np.repeat(np.array(models),8), 'Accuracy':avg_res})

x = "Model"
y = "Accuracy"
order = ['MSCNN', 'EUNet', 'MKCNN', 'XceptionTime', 'MINDS']
palette=['#66c2a5','#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
fig,ax = plt.subplots(figsize=(6,5),dpi=300,facecolor="w")
ax = sns.boxplot(data=df_res, x=x, y=y, order=order,ax=ax, palette=palette)

pairs=[("MSCNN", "MINDS"), ("EUNet", "MINDS"), ("MKCNN", "MINDS"), ("XceptionTime", "MINDS")]
annotator = Annotator(ax, pairs, data=df_res, x=x, y=y, order=order)
#annotator.configure(test='Mann-Whitney', text_format='star',line_height=0.03,line_width=1)
annotator.configure(test='Wilcoxon', text_format='star',line_height=0.03,line_width=1)
annotator.apply_and_annotate()

ax.tick_params(which='major',direction='in',length=3,width=1.,labelsize=10,bottom=False)
for spine in ["top","left","right"]:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.grid(axis='y',ls='--',c='gray')
ax.set_axisbelow(True)

plt.savefig('figs/Wilcoxon_EMG.jpg')
#plt.savefig('figs/Wilcoxon_EMG.pdf', dpi=600)

