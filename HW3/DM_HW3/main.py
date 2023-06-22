import pandas as pd
import numpy as np
import math
import random
def read_csv(file)->pd.DataFrame:
    # read csv file from pandas
    df = pd.read_csv(file)
    # convert to numpy array 
    df = df.to_numpy()
    return df
def cosine_similarity(x,y)-> int : 
    # calculate cosine similatity of two point
    size_x=x[0]**2+x[1]**2
    size_y=y[0]**2+y[1]**2
    numarator=x[0]*y[0]+x[1]*y[1]
    return numarator / (math.sqrt(size_x)*math.sqrt(size_y))

def eucilidean_distance(x,y)->int:
    # calculate euclidiean distance of two point 
    return math.sqrt(sum( (x[i]-y[i])**2 for i in range(2)))
    
def similarity_vector (dataset,type=None)-> np.array :
    lentgh = len(dataset)
    sim_vector=np.zeros([lentgh,lentgh] , dtype=float)
    for i in range(lentgh):
        for j in range (lentgh):
            if(type == "cosine"):
                sim_vector[i][j] = cosine_similarity(dataset[i],dataset[j])
            else:
                sim_vector[i][j]=eucilidean_distance(dataset[i],dataset[j])
    return sim_vector    

def should_stop(data_set,old,new,iteration):
    if (old is None):
        return False
    if iteration > 400 :
        return True
    sum=0
    for i,j in zip(old,new):
        distance= eucilidean_distance(data_set[i],data_set[j])
        sum += distance
    if distance < 1:
        return True
    
    return False
    


def get_random_centroids(lentgh,k):
    centroids=[]
    for i in range(k):
        centroids.append(random.randint(0,lentgh))
    return centroids
    
def kmeans (data_set,k):
    centroids=get_random_centroids(len(data_set),4)
    iteration = 0
    old_centroid= 
    while not should_stop (data_set,old_centroid,centroids,iteration):
        print('salam')
        old_centroid = centroids
        iteration +=1
None
    
df=read_csv('data.csv')
sim_vec_cosine=similarity_vector(df,"cosine")
sim_vec_euc=similarity_vector(df)

centroids=get_random_centroids(len(df),4)
kmeans(df,4)
print(centroids)
print(sim_vec_euc[60][60])

    