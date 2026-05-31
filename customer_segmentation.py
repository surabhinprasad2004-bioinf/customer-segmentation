import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Sample customer data
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8],
    'AmountSpent': [200, 250, 3000, 3200, 150, 180, 3500, 3300],
    'PurchaseFrequency': [2, 3, 25, 28, 1, 2, 30, 27]
}

df = pd.DataFrame(data)

# Features for clustering
X = df[['AmountSpent', 'PurchaseFrequency']]

# K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Silhouette Score
score = silhouette_score(X, df['Cluster'])

print("Customer Segmentation Results")
print("\nCustomer Clusters:")
print(df)

print("\nSilhouette Score:", round(score, 3))