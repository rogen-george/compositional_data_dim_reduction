#  Dimensionality reduction for Compositional Data
 
Comparison of different dimensionality reduction techniques for compositional data techniques for different data sets.

We try CLR and ILR transformations along with PCA, Dirichilet Correlation Analysis and CoDa AE techniques.

# 20 News Groups

The two classes 'misc.forsale' and 'alt.atheism' are selected for dimensionality reduction and classification. The data is loaded from sklearn library.

We start from sample size 10 and slowly increase the sample size and observe the performance as sample size increases. The 1425 dimension dataset is obtained after specifying a minimum frequency of 10 and removing stop words. This data is reduced to 20 dimensions using the three techniques and is classified using a linear SVM.

# Atlas Data set

The  HIT  Atlas  Dataset reported  the  proportion  of  130  genus-like groups that were present in the human intestine.  This data was collected from thefecal microbial samples of 1006 adults aged 17 - 77 from 15 different western countries. The data was obtained from this source https://bitbucket.org/RichardNock/coda/src/master/

# Diet Swap Data set

volunteers aged 50 - 65years selected from African Americans in Pittsburgh and rural native South Africans.The data consisted of proportion of different bacteria species present in the human gut.The data set contains genus level microbes whose pair wise correlations had a significantchange (>1 ) from control to treatment.  Each row was represented as a 39 dimensionalvector which summed up to 1.

The data set was obtained from here https://bitbucket.org/RichardNock/coda/src/master/

# Running the code 

Create the conda environment for the project with 

 $ conda coda_env create

This will create a conda environment with the requirements from coda_env.yml. Now run driver.py.

driver.py will gradually increase the training sample size and will store the accuracy and area under roc value for different techniques. The data will be stored as a pickle file which can be used for plotting. 

