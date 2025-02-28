import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stat_for_plot = {
    "round": [],
    "length of dataset": [],
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": [],
}

stat_folder = 'Result/Pair VLM Exp on phi newbenchmark/1740621448/stat/'

stat_files = os.listdir(stat_folder)
if stat_files:
    stat_files = [f for f in stat_files if os.path.isdir(os.path.join(stat_folder, f))]

rounds = sorted(stat_files)

count = 0
for r in rounds:
    csv_files = os.listdir(os.path.join(stat_folder, r))
    csv_files = [f for f in csv_files if f.endswith('stat.csv')]
    csv_files = sorted(csv_files)

    if count >= 3:
        break
    count += 1

    for csv_file in csv_files:
        csv_path = os.path.join(stat_folder, r, csv_file)
        df = pd.read_csv(csv_path)

        length_of_dataset = df["length of dataset"].mean()
        length_of_dataset = np.round(length_of_dataset * (1 - df["ask more questions percentage"].mean() / 100))

        stat_for_plot["round"].append(r)
        stat_for_plot["length of dataset"].append(length_of_dataset)
        stat_for_plot["accuracy"].append(df["accuracy"].mean())
        stat_for_plot["precision"].append(df["precision"].mean())
        stat_for_plot["recall"].append(df["recall"].mean())
        stat_for_plot["f1_score"].append(df["f1_score"].mean())

df = pd.DataFrame(stat_for_plot)

fig, ax1 = plt.subplots()

ax1.set_xlabel("round")
ax1.set_ylabel("metrics")

colors = plt.cm.tab10(np.linspace(0, 1, 4))  # Using Pastel1 colormap

ax1.plot(df["round"], df["accuracy"], label="accuracy", color=colors[0], linestyle='-', linewidth=3)
ax1.plot(df["round"], df["precision"], label="precision", color=colors[1], linestyle='--', linewidth=3)
ax1.plot(df["round"], df["recall"], label="recall", color=colors[2], linestyle='-.', linewidth=3)
ax1.plot(df["round"], df["f1_score"], label="f1", color=colors[3], linestyle=':', linewidth=3)

ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.set_ylabel("length of dataset")
ax2.bar(df["round"], df["length of dataset"], alpha=0.3, label="length of dataset", color="gray")
ax2.tick_params(axis='y')

fig.tight_layout()
fig.legend(loc='center right')

plt.savefig("./Visual/fig/exp_pair_phi_metric_by_round_max_3.pdf")
