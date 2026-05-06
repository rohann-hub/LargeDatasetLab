import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("census_sample.csv")
data_encoder = pd.get_dummies(data)
x = data_encoder.values
scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)
print("Explained Variance :", pca.explained_variance_ratio_)   
print(x_pca[:5]) 