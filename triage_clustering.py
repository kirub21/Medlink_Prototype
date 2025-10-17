```python
# Install required libraries
!pip install pandas scikit-learn matplotlib kagglehub joblib

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import kagglehub
import joblib

# Download the dataset
path = kagglehub.dataset_download("prasad22/healthcare-dataset")
print("Path to dataset files:", path)

# Step 1: Load and prepare data
df = pd.read_csv(path + '/healthcare_dataset.csv')

# Convert dates to datetime and calculate Length of Stay
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

# Select features
features_df = df[['Age', 'Gender', 'Admission Type', 'Medical Condition', 'Length of Stay', 'Billing Amount']]
features_df = features_df.dropna()

# Define numeric and categorical features
numeric_features = ['Age', 'Length of Stay', 'Billing Amount']
categorical_features = ['Gender', 'Admission Type', 'Medical Condition']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Apply preprocessing
processed_features = preprocessor.fit_transform(features_df)

# Step 2: Determine optimal number of clusters (Elbow Method)
inertias = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(processed_features)
    inertias.append(kmeans.inertia_)

# Plot Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.savefig('/content/elbow_plot.png')
plt.show()

# Choose k (adjust based on elbow plot; default to 3)
optimal_k = 3

# Step 3: Fit K-Means model
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(processed_features)

# Evaluate with Silhouette Score
sil_score = silhouette_score(processed_features, clusters)
print(f'Silhouette Score for k={optimal_k}: {sil_score:.3f}')

# Step 4: Visualize clusters (using PCA)
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(processed_features.toarray() if hasattr(processed_features, 'toarray') else processed_features)

plt.figure(figsize=(8, 5))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clusters, cmap='viridis')
plt.title('Patient Clusters Visualization (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.savefig('/content/cluster_visualization.png')
plt.show()

# Step 5: Interpret clusters
features_df['cluster'] = clusters
cluster_summary = features_df.groupby('cluster').mean(numeric_only=True)
print('Cluster Summary (Averages):')
print(cluster_summary)
cluster_summary.to_csv('/content/cluster_summary.csv')

# Example interpretation
print('Example Interpretation:')
print('- Cluster 0: Low values – Low-risk, routine care')
print('- Cluster 1: Moderate – Monitor remotely')
print('- Cluster 2: High values – Urgent triage in rural clinic')

# Step 6: Save the model and preprocessor
joblib.dump(kmeans, '/content/kmeans_model.pkl')
joblib.dump(preprocessor, '/content/preprocessor.pkl')
print('Model and preprocessor saved as /content/kmeans_model.pkl and /content/preprocessor.pkl')

# Download output files
from google.colab import files
files.download('/content/elbow_plot.png')
files.download('/content/cluster_visualization.png')
files.download('/content/cluster_summary.csv')
files.download('/content/kmeans_model.pkl')
files.download('/content/preprocessor.pkl')
```