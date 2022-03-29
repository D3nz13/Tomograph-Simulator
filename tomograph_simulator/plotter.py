from cProfile import label
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os import path

sns.set_theme(style='darkgrid')

fig, axs = plt.subplots(nrows=3, figsize=(8,12))

csvs = [f for f in listdir("./") if path.isfile(path.join("./", f)) and '.csv' in f]

for csv in csvs:
    df = pd.read_csv(csv)
    csv_label = csv.split(".")[0][8:]
    sns.lineplot(data=df[df["Group"] == "Detectors"], x="Detectors", y="RMSE", label=csv_label, ax=axs[0])
    sns.lineplot(data=df[df["Group"] == "Iterations"], x="Iterations", y="RMSE", label=csv_label, ax=axs[1])
    sns.lineplot(data=df[df["Group"] == "Span"], x="Span", y="RMSE", label=csv_label, ax=axs[2])

plt.savefig("plot.png")