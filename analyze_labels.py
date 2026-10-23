import numpy as np
import matplotlib.pyplot as plt

label_names = [
    "Center",
    "Donut",
    "Edge_Loc",
    "Edge_Ring",
    "Loc",
    "Near_Full",
    "Scratch",
    "Random"
]

with np.load("Wafer_Map_Datasets.npz") as data:
    labels = data["arr_1"]

label_counts = labels.sum(axis=0)

for name, count in zip(label_names, label_counts):
    print(
        f"{name}: {int(count)} wafers "
        f"({count / len(labels) * 100:.2f}%)"
    )

plt.figure(figsize=(11, 6))
plt.bar(label_names, label_counts)

plt.xlabel("Defect Type")
plt.ylabel("Number of Wafers")
plt.title("Wafer Defect Label Distribution")
plt.xticks(rotation=35, ha="right")

plt.tight_layout()
plt.savefig("label_distribution.png", dpi=150)
plt.show()