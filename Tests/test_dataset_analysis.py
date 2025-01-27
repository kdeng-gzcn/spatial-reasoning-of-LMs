import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import sys
sys.path.append("./")

done = 1

if not done:

    df_25 = pd.read_csv("./data/RGBD_7_Scenes_25/RGBD_7_Scenes_25.csv")
    df_50 = pd.read_csv("./data/RGBD_7_Scenes_Custom/RGBD_7_Scenes_50.csv")
    df_100 = pd.read_csv("./data/RGBD_7_Scenes_100/RGBD_7_Scenes_100.csv")

    grouped_25 = df_25.groupby(['scene', 'seq'])
    grouped_50 = df_50.groupby(['scene', 'seq'])
    grouped_100 = df_100.groupby(['scene', 'seq'])

    grouped_keys = set(grouped_50.groups.keys())

    for (scene, seq) in grouped_keys:

        group_data_25 = grouped_25.get_group((scene, seq)) if (scene, seq) in grouped_25.groups else None
        group_data_50 = grouped_50.get_group((scene, seq)) if (scene, seq) in grouped_50.groups else None
        group_data_100 = grouped_100.get_group((scene, seq)) if (scene, seq) in grouped_100.groups else None

        scene_dir = f"./results/dataset/hist_abs/{scene}"
        os.makedirs(scene_dir, exist_ok=True)

        fig, axes = plt.subplots(3, 2, figsize=(18, 18), sharey=False)
        fig.suptitle(f"Scene: {scene}, Seq: {seq}, (tx, ty, tz, theta, phi, psi)", fontsize=18)

        translation_cols = ['t_x', 't_y', 't_z']
        rotation_cols = ['theta', 'phi', 'psi']

        datasets = [(group_data_25, 'Frame 25'), (group_data_50, 'Frame 50'), (group_data_100, 'Frame 100')]

        for i, (data, label) in enumerate(datasets):
            if data is not None:

                axes[i, 0].set_title(f'{label} - Translation', fontsize=14)
                for col in translation_cols:
                    sns.kdeplot(data[col].abs(), label=col, fill=True, alpha=0.5, ax=axes[i, 0])

                axes[i, 0].set_xlabel('Translation Values')
                axes[i, 0].set_ylabel('Density')
                axes[i, 0].legend()
                axes[i, 0].grid()

                axes[i, 1].set_title(f'{label} - Rotation', fontsize=14)
                for col in rotation_cols:
                    sns.kdeplot(data[col].abs(), label=col, fill=True, alpha=0.5, ax=axes[i, 1])

                axes[i, 1].set_xlabel('Rotation Values')
                axes[i, 1].legend()
                axes[i, 1].grid()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_path = f"{scene_dir}/hist_{scene}_{seq}.pdf"
        plt.savefig(output_path)
        plt.close()

        print(f"Saved: {output_path}")

done = 0

if not done:

    pass
