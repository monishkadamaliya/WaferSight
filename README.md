# WaferSight — Wafer Defect Intelligence

WaferSight is a Streamlit-based application for exploring semiconductor wafer maps and viewing machine-learning predictions for wafer defect labels. It combines wafer-map visualization with multi-label prediction scores and, when labels are available in the dataset, a comparison against the sample's ground-truth labels.


## Ownership & Copyright

**WaferSight** is an independently developed machine-learning project created by **Monish Jain**.

Copyright © 2026 Monish Jain. All rights reserved, subject to applicable third-party rights and licenses.

The original source code, project-specific documentation, and original application design are protected by applicable copyright law to the extent they qualify for protection. No license is granted to reproduce, redistribute, or commercially exploit these original materials except as permitted by applicable law or with prior written permission from the relevant rights holder.

Third-party libraries, datasets, model artifacts, and other materials remain subject to their respective owners' rights and applicable licenses. This notice does not claim ownership of third-party materials or of the underlying wafer-map dataset.

**Official project links**
- **Creator:** Monish Jain
- **Live demo:** https://wafersight-monishjain.streamlit.app/
- **Source repository:** https://github.com/monishkadamaliya/WaferSight
- **Dataset repository:** https://huggingface.co/datasets/monishkadamaliya/wafersight-wafer-maps

## Features

- **Wafer-map visualization:** Inspect an individual wafer map in the dashboard.
- **Multi-label predictions:** View predicted scores for the supported defect labels.
- **Threshold-based decisions:** Convert model scores into predicted labels using the thresholds stored with the model artifact.
- **Ground-truth comparison:** Compare predictions with dataset labels for the selected sample.
- **Interactive sample selection:** Browse different samples from the included dataset.
- **Streamlit interface:** Run the dashboard locally in a web browser.

## Project files

Keep these files in the same project directory unless you update the paths in `app.py`:

```text
WaferSight/
├── app.py
├── wafer_random_forest.joblib
├── Wafer_Map_Datasets.npz
├── requirements.txt
└── README.md
```

The `.venv-demo/` directory may also exist locally as your Python virtual environment. It generally should **not** be committed to Git.

- `app.py` — Streamlit dashboard.
- `wafer_random_forest.joblib` — Saved Random Forest model artifact, including the model metadata expected by the app.
- `Wafer_Map_Datasets.npz` — Dataset archive. The app expects arrays named `arr_0` (wafer maps) and `arr_1` (labels).
- `requirements.txt` — Python package dependencies.
- `README.md` — Project documentation.

> **Data and artifact note:** The model and dataset files may be large or subject to sharing restrictions. Only distribute them if you have permission, and avoid committing sensitive or restricted data to a public repository.

## Requirements

- Python 3.10 or newer is recommended.
- `pip`
- The project files listed above.

The application uses Streamlit, NumPy, pandas, Matplotlib, Joblib, and scikit-learn. See [`requirements.txt`](requirements.txt) for the dependency list.

## Installation

### 1. Open a terminal in the project folder

For example:

```powershell
cd path\to\WaferSight
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -m venv .venv-demo
.\.venv-demo\Scripts\Activate.ps1
```

If PowerShell prevents activation because of the execution policy, use an appropriate environment-specific workaround or activate the environment from Command Prompt instead:

```bat
.venv-demo\Scripts\activate.bat
```

**macOS / Linux:**

```bash
python3 -m venv .venv-demo
source .venv-demo/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Add the model and dataset

Make sure `wafer_random_forest.joblib` and `Wafer_Map_Datasets.npz` are in the same directory as `app.py`, unless you have changed the file paths in the application.

## Run WaferSight

From the project directory, with the virtual environment activated, run:

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal, typically `http://localhost:8501`. Open that address in your browser. To stop the server, press `Ctrl+C` in the terminal.

## Using the dashboard

1. Launch the app using the command above.
2. Select a wafer-map sample in the sidebar.
3. Inspect the wafer map and its visual legend.
4. Run the prediction action to view the model's defect scores and predicted labels.
5. If ground-truth labels are present, review the comparison to understand where the model agrees or disagrees with the dataset labels.
6. Try several samples rather than drawing conclusions from one prediction.

## Model and evaluation

WaferSight uses a saved Random Forest model for multi-label wafer-defect prediction. The app loads the saved artifact and uses its stored thresholds to turn per-label probabilities into predicted labels.

Record the verified evaluation results from your held-out test set here before publishing the project. Do not report training-set scores as test performance.

| Metric | Verified result |
|---|---|
| Macro F1 | Add your held-out test result |
| Micro F1 | Add your held-out test result |
| Hamming loss | Add your held-out test result |

For a more complete evaluation, include per-label precision, recall, and F1 scores, and describe how the test split was created. Make sure threshold selection did not use the held-out test labels.

## Important limitations

- Predictions are model outputs, not a guarantee that a wafer has or does not have a defect.
- Prediction scores and thresholded labels depend on the saved model, preprocessing, and thresholds.
- Results should be interpreted alongside manufacturing context and appropriate expert review.
- Performance can change when the input data differs from the data used to train and evaluate the model.
- The dashboard's sample-level ground-truth comparison is only meaningful when the dataset labels are reliable and correctly aligned with the wafer maps.

## Troubleshooting

**The app says a file is missing**
- Check that `app.py`, `wafer_random_forest.joblib`, and `Wafer_Map_Datasets.npz` are in the expected directory.
- If you renamed a file or moved it, update the corresponding path in `app.py`.

**`streamlit` is not recognized**
- Activate the virtual environment and run `python -m pip install -r requirements.txt`.
- Start the app with `python -m streamlit run app.py`.

**A package import fails**
- Confirm that the correct virtual environment is active.
- Reinstall dependencies with `python -m pip install -r requirements.txt`.

**The model artifact cannot be loaded**
- Confirm that the artifact was created with a compatible version of scikit-learn and that the file is not corrupted. Joblib/pickle-based artifacts should only be loaded from trusted sources.

## Reproducibility and security

- Keep a record of the Python and package versions used for training and evaluation.
- If exact reproducibility is required, pin the dependency versions from the environment in which the model was trained and tested.
- Only load Joblib artifacts from trusted sources; deserialization can execute code.
- Do not commit `.venv-demo/`, temporary files, or restricted datasets to version control.

## License and data attribution

No permissive open-source license is granted by this repository's ownership notice. Unless a separate license says otherwise, the original project materials remain subject to applicable copyright law. Public visibility on GitHub does not mean the project is in the public domain; GitHub's platform terms may still allow viewing and forking on GitHub.

The dataset and third-party dependencies are not claimed as original works by the project author. Check the original dataset terms and each dependency's license before redistribution or commercial use.

## Author

Created by **Monish Jain**.

- Live demo: https://wafersight-monishjain.streamlit.app/
- GitHub: https://github.com/monishkadamaliya/WaferSight
