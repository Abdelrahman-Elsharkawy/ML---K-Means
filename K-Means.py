#Import Used Libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


#Reading Dataset
data = pd.read_csv('data.csv')
data


#Plot The Data
plt.scatter(data['Longitude'],data['Latitude'])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()


#Selecting Features by Splitting Dataset
x = data.iloc[:,1:3] # 1t for rows and second for columns
x


#Clustering
kmeans = KMeans(3)
kmeans.fit(x)


#Clustering Result
identified_clusters = kmeans.fit_predict(x)
identified_clusters
data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['Longitude'],data_with_clusters['Latitude'],c=data_with_clusters['Clusters'],cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],color='y')

#WCSS and Elbow Method to Find the Best no of Clusters 
wcss=[]
for i in range(1,7):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

number_clusters = range(1,7)
plt.plot(number_clusters,wcss)
plt.title('The Elbow title')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
