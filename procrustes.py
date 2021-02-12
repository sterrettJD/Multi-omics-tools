import pandas as pd
import numpy as np
from scipy.spatial import procrustes

#A procrustes randomization test! aka protest
def protest(a,b,n=999):
  """a: pd.DataFrame of Ordination coordinates for dataset a
     b: pd.DataFrame of Ordination coordinates for dataset b
     n: Integer, number of randomizations
     - - - - - - - - - - - - - - - - - - - - 
     returns:
     disparity, pval"""
  mtx1, mtx2, disparity = procrustes(a, b)
  
  #Randomization test time
  rows, cols = a.shape

  #disparities will be the list containing M^2 values for each test
  disparities = []
  for x in range(n):

      #start by randomly sampling 100% of the ordination coordinates
      a_rand = a.sample(frac=1,axis=0).reset_index(drop=True)
      b_rand = b.sample(frac=1,axis=0).reset_index(drop=True)

      #run the procrustes on the two randomly sampled coordinates
      mtx1_rand, mtx2_rand, disparity_rand = procrustes(a_rand,
                                                        b_rand)

      #add the result to disparities
      disparities.append(disparity_rand)

  #set pval = proportion of randomized samples where the random disparity is smaller than
  #our observed disparity
  pval = (sum([disparity > d for d in disparities])+1)/(n+1)
  
  return disparity, pval, disparities

#EXAMPLE
a = pd.DataFrame(np.array([[1, 3], 
                           [1, 2], 
                           [1, 1], 
                           [2, 1]]))
b = pd.DataFrame(np.array([[4, -2], 
                           [4, -4], 
                           [4, -6], 
                           [2, -6]]))

disp, p, disps = protest(a,b)

print(f'pval: {p} \ndisparity: {disp}')
