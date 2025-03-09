import os

import pandas as pd
import matplotlib.pyplot as plt

benchmark_dir = "./benchmark/Rebuild_7_Scenes_1739853799/global_metadata.csv"

# for subset in os.listdir(benchmark_dir):
#     subset_path = benchmark_dir / subset
#     if not os.path.isdir(subset_path):
#         continue

#     for scene in os.listdir(subset_path):
#         scene_path = subset_path / scene
#         if not os.path.isdir(scene_path):
#             continue

#         for seq in os.listdir(scene_path):
#             seq_path = scene_path / seq
#             if not os.path.isdir(seq_path):
#                 continue

#             pairs = os.listdir(seq_path)

df = pd.read_csv(benchmark_dir, header=0)

significance_counts = df['significance'].value_counts()
plt.figure(figsize=(9, 9))
plt.pie(significance_counts, labels=significance_counts.index, autopct='%1.1f%%', startangle=90)
plt.title(f'Significance Distribution (Total Length: {len(df)})')
plt.axis('equal')
plt.savefig("Significante Dist.pdf")

phi_scene_counts = df[df['significance'] == 'phi']['scene'].value_counts()

phi_scene_counts
plt.figure(figsize=(9, 9))
plt.pie(phi_scene_counts, labels=phi_scene_counts.index, autopct='%1.1f%%', startangle=90)
plt.title(f'Scene Distribution in direction Phi (Total Length: {phi_scene_counts.sum()})')
plt.axis('equal')
plt.savefig("Scene Dist.pdf")

# # Plot a horizontal stacked bar chart instead
# ax = df_new.plot(kind='barh', stacked=True, figsize=(10, 6), color=['skyblue', 'salmon', 'lightgreen'])

# # Set the title and labels
# plt.title('Confusion Matrix - Horizontal Stacked Bar Chart', fontsize=16)
# plt.xlabel('Count', fontsize=12)
# plt.ylabel('Predicted Category', fontsize=12)
# plt.legend(title='True Category', title_fontsize='13', fontsize='11')
# plt.tight_layout()

# # Show the plot
# plt.show()
