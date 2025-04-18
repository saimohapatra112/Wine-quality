import pandas as pd
df = pd.read_csv(r"wine.data.csv")
df.head()

df.shape

import warnings
warnings.filterwarnings('ignore')

df.isnull().sum()

df.dtypes

from sklearn.preprocessing import StandardScaler
x = df.iloc[:,1:]
y = df.iloc[:,0]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

from sklearn.decomposition import PCA

#apply PCA
pca = PCA()
x_pca = pca.fit_transform(x_scaled)

#explained variance
explained_variance = pca.explained_variance_ratio_


pca_2 = PCA(n_components=2)
x_pca_2 = pca_2.fit_transform(x_scaled)
pca_df = pd.DataFrame(data = x_pca_2, columns = ['PC1','PC2'])

x=pd.DataFrame(pca_df)

x.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

knn.score(x_test,y_test)
import joblib

# Save the trained model and PCA
joblib.dump(knn, "knn_model.pkl")
joblib.dump(pca_2, "pca_model.pkl")
joblib.dump(scaler, "scaler_model.pkl")
