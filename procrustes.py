import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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


def procrustes_plot(a,b, a_name,b_name):
    """a: ordination coordinates for dataset a
     b: ordination coordinates for dataset b
     a_name: name for dataset a
     b_name: name for dataset b
     - - - - - - - - - - - - - - - - - - - - 
     returns:
     ax: axes of plot
     disp: disparity score of procrustes
    """

    # do procrustes
    mtx1, mtx2, disparity = procrustes(a, b)

    # Make our plotting df
    proplot = pd.concat([pd.DataFrame(mtx1), pd.DataFrame(mtx2)])
    proplot.columns = ["PCo1", "PCo2", "PCo3"]
    proplot["Dataset"] = [a_name]*a.shape[0] + [b_name]*b.shape[0]
    
    # plot
    ax = sns.scatterplot(x="PCo1",y="PCo2",
                         style="Dataset",hue="Dataset",
                         data=proplot,
                         markers=["v","o"],
                         s=150)
    # Add lines
    for i in range(len(mtx1)):
        plt.plot([mtx1[i,0],mtx2[i,0]],
                 [mtx1[i,1],mtx2[i,1]],
                 c="black", linewidth=0.75)

    return ax, disparity


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
