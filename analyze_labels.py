
import numpy as np
import matplotlib.pyplot as plt

with np.load("Wafer_Map_Datasets.npz") as data:
    labels = data["arr_1"]

label_counts = labels.sum(axis=0)

for i, count in enumerate(label_counts):
    print(f"Label {i}: {count} wafers ({count / len(labels) * 100:.2f}%)")

plt.figure(figsize=(9, 5))
plt.bar(range(labels.shape[1]), label_counts)
plt.xlabel("Label index")
plt.ylabel("Number of wafers")
plt.title("Wafer Defect Label Distribution")
plt.xticks(range(labels.shape[1]))
plt.tight_layout()
plt.savefig("label_distribution.png", dpi=150)
plt.show()
