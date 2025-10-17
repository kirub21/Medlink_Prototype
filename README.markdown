# Healthcare Triage ML Assignment: Unsupervised Clustering and Decision Support for Rural Areas (Colab)

## Overview

This project implements an unsupervised machine learning solution for healthcare triage in rural areas using K-Means clustering, optimized for Google Colab. It consists of two scripts:

1. **Triage Clustering** (`triage_clustering.py`): Clusters patients based on features like age, gender, admission type, medical condition, length of stay, and billing amount to identify risk levels.
2. **Decision Support** (`decision_support.py`): Provides an interactive `ipywidgets` interface to input new patient data and predict their risk cluster.

## Dataset

- **Source**: Kaggle dataset `prasad22/healthcare-dataset` (https://www.kaggle.com/datasets/prasad22/healthcare-dataset).
- **Description**: \~10,000 hospital records with columns: Name, Age, Gender, Blood Type, Medical Condition, Date of Admission, Doctor, Hospital, Insurance Provider, Billing Amount, Room Number, Admission Type, Discharge Date, Medication, Test Results.
- **Usage**: Features used: Age, Gender, Admission Type, Medical Condition, Length of Stay (derived), Billing Amount.

## Requirements

- **Environment**: Google Colab (no local Python installation needed).
- **Libraries**: Installed automatically in scripts:
  - pandas
  - scikit-learn
  - matplotlib
  - kagglehub
  - joblib
  - ipywidgets

## Setup

1. Open Google Colab (https://colab.research.google.com).
2. Upload or create `triage_clustering.py` and `decision_support.py`:
   - Click `File > New Notebook`.
   - Copy-paste each script into separate notebooks or cells.
   - Alternatively, save as `.py` files and upload via `File > Upload Notebook` or drag-and-drop.
3. Set up Kaggle API for dataset download:
   - Go to https://www.kaggle.com/settings, create an API token, and download `kaggle.json`.
   - In Colab, upload `kaggle.json`:

     ```python
     from google.colab import files
     files.upload()  # Upload kaggle.json
     !mkdir -p ~/.kaggle
     !cp kaggle.json ~/.kaggle/
     !chmod 600 ~/.kaggle/kaggle.json
     ```
4. Ensure internet is enabled in Colab (`Settings > Runtime > Internet on`).

## Usage

### 1. Triage Clustering (`triage_clustering.py`)

- **Run**: Execute all cells in the notebook containing `triage_clustering.py`.
- **Outputs** (saved to `/content/`):
  - `elbow_plot.png`: Elbow curve for optimal k.
  - `cluster_visualization.png`: PCA scatter plot of clusters.
  - `cluster_summary.csv`: Average feature values per cluster.
  - `kmeans_model.pkl`: Trained K-Means model.
  - `preprocessor.pkl`: Preprocessor for feature encoding/scaling.
  - Silhouette score (printed; &gt;0.5 indicates good clustering).
- **Download Outputs**: Files are automatically downloaded via `files.download()`. Check your local Downloads folder.
- **Interpretation**: Clusters represent risk levels (e.g., Cluster 0: Low-risk, Cluster 2: Urgent).


<img width="800" height="500" alt="cluster_visualization" src="https://github.com/user-attachments/assets/43935a8d-5735-451b-8aa6-448fcd6ca81e" />

### 2. Decision Support (`decision_support.py`)

- **Run**: Execute after `triage_clustering.py` in a new notebook, ensuring `.pkl` files are in `/content/`.
- **Interface**: An `ipywidgets` interface with:
  - Age (numeric input)
  - Gender (dropdown: Male/Female)
  - Admission Type (dropdown: Emergency/Elective/Urgent)
  - Medical Condition (dropdown: Diabetes/Hypertension/Asthma/etc.)
  - Length of Stay (numeric input)
  - Billing Amount (numeric input)
- **Usage**:
  1. Fill in patient details (e.g., Age: 45, Gender: Male, Admission Type: Emergency).
  2. Click “Predict Cluster” to see the result (e.g., “Predicted Cluster: 2 (Urgent)”).
- **Output**: Predicted cluster displayed below the button.
<img width="417" height="360" alt="image" src="https://github.com/user-attachments/assets/d2275483-a693-418c-85e5-3af3e4eb3dd5" />

## Results Example

- **Optimal k**: 3 clusters (low/medium/high risk).
- **Silhouette Score**: \~0.5–0.6 (adjust k if needed).
- **Clusters**: E.g., high billing amount and emergency admissions indicate urgent cases.
- **Decision Support**: Sample patient (Age=45, Male, Emergency, Diabetes, 5 days, $25,000) assigned to a cluster.

## Limitations

- Dataset lacks rural-specific indicators; generalize cautiously.
- Missing vital signs; Length of Stay and Billing Amount proxy severity.
- Unsupervised clustering requires domain validation.
- Ensure `.pkl` files are available for `decision_support.py`.

## References

- Dataset: https://www.kaggle.com/datasets/prasad22/healthcare-dataset
- scikit-learn: https://scikit-learn.org/stable/modules/clustering.html
- ipywidgets: https://ipywidgets.readthedocs.io
- KaggleHub: https://github.com/Kaggle/kagglehub

## Notes

- Run `triage_clustering.py` first to generate `.pkl` files.
- For rural focus, consider DHS datasets (https://dhsprogram.com/data/).
- Validate clusters with healthcare experts.

**Author**: kirubel Gulilat | [Medlink](https://www.medlinket.et/)  
**Date**: October 17, 2025
