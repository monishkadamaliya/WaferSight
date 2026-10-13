import numpy as np
from pathlib import Path

# Find .npz datasets in the project folder
files = list(Path(".").glob("*.npz"))

if not files:
	print("No .npz file found in this folder.")
	print("Place your wafer dataset in the project folder.")
else:
	dataset_path = files[0]
	print("Dataset:", dataset_path.name)

	with np.load(dataset_path, allow_pickle=False) as data:
		print("\nAvailable arrays:", data.files)

		for name in data.files:
			arr = data[name]
			print(f"\nArray: {name}")
			print("Shape:", arr.shape)
			print("Data type:", arr.dtype)

			if arr.size > 0:
				print("First sample shape:", arr[0].shape)
				print("First sample values:", np.unique(arr[0])[:20])
