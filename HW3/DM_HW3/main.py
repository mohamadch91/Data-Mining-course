import pandas as pd
import numpy as np
import math
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
    
def similarity_vector (dataset,type)-> np.array :
    lentgh = len(dataset)
    sim_vector=np.zeros([lentgh,lentgh] , dtype=float)
    for i in range(lentgh):
        for j in range (lentgh):
            if(type == "cosine"):
                sim_vector[i][j] = cosine_similarity(dataset[i],dataset[j])
            else:
                sim_vector[i][j]=eucilidean_distance(dataset[i],dataset[j])
    return sim_vector    
    
    
df=read_csv('data.csv')
sim_vec=similarity_vector(df,"cosine")
print(sim_vec[60][60])

    