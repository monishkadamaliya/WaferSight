
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
with np.load("Wafer_Map_Datasets.npz") as data:
    wafers = data["arr_0"]
    labels = data["arr_1"]

# Show the first 8 wafer maps
fig, axes = plt.subplots(2, 4, figsize=(14, 7))

for i, ax in enumerate(axes.flat):
    ax.imshow(wafers[i], cmap="viridis", vmin=0, vmax=2)
    active_labels = np.where(labels[i] == 1)[0]

    ax.set_title(f"Wafer {i}\nActive labels: {active_labels.tolist()}")
    ax.axis("off")

plt.suptitle("Wafer Map Defect Samples", fontsize=16)
plt.tight_layout()
plt.savefig("wafer_samples.png", dpi=150, bbox_inches="tight")
plt.show()

print("Wafer maps:", wafers.shape)
print("Label matrix:", labels.shape)
print("Saved visualization to wafer_samples.png")
