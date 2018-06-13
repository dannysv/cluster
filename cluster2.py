from sklearn.cluster import KMeans
from numpy import genfromtxt
import numpy as np

class Cluster():
    def __init__(self, k, path_csv):
        self.k = k
        self.path_csv = path_csv

    def get_id_data(self):
        csv_ = genfromtxt(self.path_csv, delimiter=',', skip_header=1)
        self.ids = csv_[:,0]
        self.data = csv_[:,1:]

    def get_kmeans(self):
        self.kmeans = KMeans(n_clusters=self.k, random_state=0).fit(self.data)
        self.labels = self.kmeans.labels_
        self.cluster_centers_ = self.kmeans.cluster_centers_

    def get_n_cluster(self, n):
        if(n<self.k):           
            resp = [cl.ids[i] for (i,a) in enumerate(self.labels) if a==n]
            return resp
        else:
            print("invalid cluster index")

    #we can add methods to cluster new elements ...


#number of clusters
k=4

#dataset
path_csv = 'dataset-kmeans.csv'

#clusterization
cl = Cluster(k, path_csv)
cl.get_id_data()
cl.get_kmeans()

#print the clustering result
for i in range(0,k):
    print("cluster nro: "+str(i))
    print(cl.get_n_cluster(i))
