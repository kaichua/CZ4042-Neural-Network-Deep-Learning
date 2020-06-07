import tensorflow as tf
import numpy as np
import pylab as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('admission_predict.csv')
df = df.drop(columns=['Serial No.'])
fig,ax = plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.subplots_adjust(bottom=0.25)
plt.show()