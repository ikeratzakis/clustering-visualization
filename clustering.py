import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import confusion_matrix, classification_report

# Read second sheet from xls file into a dataframe
df = pd.read_excel('CTG.xls', sheet_name=1, skiprows=[0])
print(df.head)
# Since last three rows contain invalid data, we delete them, and we also delete empty columns with weird names
# (Unnamed: 9, 31, 42, 44)
df = df.drop(axis=0, index=[2126, 2127, 2128])
df = df.drop(axis=1, labels=["Unnamed: 9", "Unnamed: 31", "Unnamed: 42", "Unnamed: 44"])
# Check for NaN values
print('Checking for NaN values... ', df.isnull().values.any())
# Looking good! Save dataframe to CSV for future usage
df.to_csv(path_or_buf='cardio.csv')

# Perform PCA to reduce data to two dimensions, after standardizing it
X = df.drop(['NSP', 'CLASS'], axis=1).to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X)
pca = PCA(n_components=2)
X = pca.fit_transform(X)
print('Dataset after scaling/applying PCA:', X)
# We subtract 1 from each element in each class list due to how heatmaps handle indexing (they start from 0 instead
# of 1) .
y_CLASS = [int(i - 1) for i in df['CLASS'].values.tolist()]
y_NSP = [int(i - 1) for i in df['NSP'].values.tolist()]

# Try KMeans with 10 clusters (CLASS) and 3 clusters (NSP)
print('Performing KMeans clustering')
kmeans_CLASS = KMeans(n_clusters=10)
kmeans_CLASS_predictions = kmeans_CLASS.fit_predict(X)
kmeans_NSP = KMeans(n_clusters=3)
kmeans_NSP_predictions = kmeans_NSP.fit_predict(X)
# Agglomerative clustering
print('Performing agglomerative clustering')
agglo_CLASS = AgglomerativeClustering(n_clusters=10)
agglo_CLASS_predictions = agglo_CLASS.fit_predict(X)
agglo_NSP = AgglomerativeClustering(n_clusters=3)
agglo_NSP_predictions = agglo_NSP.fit_predict(X)
# Spectral clustering
print('Performing spectral clustering')
spectral_CLASS = SpectralClustering(n_clusters=10, n_jobs=-1)
spectral_CLASS_predictions = spectral_CLASS.fit_predict(X)
spectral_NSP = SpectralClustering(n_clusters=3, n_jobs=-1)
spectral_NSP_predictions = spectral_NSP.fit_predict(X)

# Compute confusion matrices and classification reports. We will store all the variables for visualization.
kmeans_CLASS_cm = confusion_matrix(y_CLASS, kmeans_CLASS_predictions)
kmeans_NSP_cm = confusion_matrix(y_NSP, kmeans_NSP_predictions)
agglo_CLASS_cm = confusion_matrix(y_CLASS, agglo_CLASS_predictions)
agglo_NSP_cm = confusion_matrix(y_NSP, agglo_NSP_predictions)
spectral_CLASS_cm = confusion_matrix(y_CLASS, spectral_CLASS_predictions)
spectral_NSP_cm = confusion_matrix(y_NSP, spectral_NSP_predictions)

print('Classification reports for 10 class predictions (CLASS):')
for method, predictions in zip(['KMeans', 'Agglomerative Clustering', 'Spectral Clustering'], [kmeans_CLASS_predictions,
                                                                                               agglo_CLASS_predictions,
                                                                                               spectral_CLASS_predictions]):
    print('Method: ', method, '\n', classification_report(y_CLASS, predictions))

print('Classification reports for 3 class predictions (NSP):')
for method, predictions in zip(['KMeans', 'Agglomerative Clustering', 'Spectral Clustering'], [kmeans_NSP_predictions,
                                                                                               agglo_NSP_predictions,
                                                                                               spectral_NSP_predictions]):
    print('Method: ', method, '\n', classification_report(y_NSP, predictions))

# Time to visualize the clusters, along with ground truth labels
fig, ax = plt.subplots(nrows=2, ncols=4)
titles = ['KMeans', 'Agglomerative clustering (Ward)', 'Spectral clustering', 'Actual labels (ground truth)']
indices = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
colors = [kmeans_CLASS_predictions, agglo_CLASS_predictions, spectral_CLASS_predictions, y_CLASS, kmeans_NSP_predictions,
          agglo_NSP_predictions, spectral_NSP_predictions, y_NSP]
for title, index in zip(titles, indices[:4]):
    ax[index[0], index[1]].title.set_text(title)

for index, color in zip(indices, colors):
    ax[index[0], index[1]].scatter(X[:, 0], X[:, 1], c=color, s=10)

plt.tight_layout()
plt.show()

# And also visualize the confusion matrices
confusion_matrices = [kmeans_CLASS_cm, agglo_CLASS_cm, spectral_CLASS_cm, kmeans_NSP_cm, agglo_NSP_cm, spectral_NSP_cm]
cm_indices = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
fig, ax = plt.subplots(nrows=2, ncols=3)
ax[0, 0].title.set_text('KMeans confusion matrices')
ax[0, 1].title.set_text('Agglomerative clustering (Ward) confusion matrices')
ax[0, 2].title.set_text('Spectral clustering confusion matrices')
for matrix, index in zip(confusion_matrices, cm_indices):
    sns.heatmap(matrix, annot=True, ax = ax[index[0], index[1]])
plt.tight_layout()
plt.show()
