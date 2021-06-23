# 20 News Groups for Dimensionality reduction
 

Comparison of different dimensionality reduction techniques for compositional data techniques on 20 News Groups dataset.

We try CLR and ILR transformations along with PCA, Dirichilet Correlation Analysis and CoDa AE techniques on 20 News Groups data.

The two classes 'misc.forsale' and 'alt.atheism' are selected for dimensionality reduction and classification.

We start from sample size 10 and slowly increase the sample size and observe the performance as sample size increases. The 1425 dimension dataset is obtained after specifying a minimum frequency of 10 and removing stop words. This data is reduced to 20 dimensions using the three techniques and is classified using a linear SVM.

# Running the code 

Create the conda environment for the project with 

 $ conda coda_env create

This will create a conda environment with the requirements from coda_env.yml. Now run driver.py.
