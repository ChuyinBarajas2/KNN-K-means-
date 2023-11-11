
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
import os
import pandas as pd
import numpy as np

# Datos
directorio_actual = os.getcwd()
df = pd.read_excel(directorio_actual + '/Datos.xlsx')
X = df['diabetes'].values.reshape(-1, 1) 
y = df['gender'].values
#KNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy}')

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
classification_report = metrics.classification_report(y_test, y_pred)

print('Matriz de Confusión:')
print(confusion_matrix)

print('\nInforme de Clasificación:')
print(classification_report)


# K-Means
def kmeans(X, k, max_iters=10000, tol=1e-4):
    X = np.array(X)  
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]
    for _ in range(max_iters):
        distances = np.linalg.norm(X - centroids[:, np.newaxis], axis=2)
        labels = np.argmin(distances, axis=0)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.linalg.norm(new_centroids - centroids) < tol:
            break
    
        centroids = new_centroids

    return labels, centroids

# K-Means 80/20
split_idx = int(0.8 * len(X))
train_data, test_data = X[:split_idx], X[split_idx:]

k = 4
labels, centroids = kmeans(train_data, k)

print("Etiquetas K-Means:", labels)
print("Centroides finales K-Means:", centroids)