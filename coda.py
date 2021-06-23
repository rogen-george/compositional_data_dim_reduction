import numpy as np
from CodaPCA import Alg, PCA, TSNE, CLRPCA, CodaPCA, NonParametricCodaPCA, clr_transform
import numpy as np
import pandas as pd
#
#
# PCA = 1
# TSNE = 2
#
# CLRPCA = 3
# CLRPCANN = 4
#
# CODAPCA = 5
# SCODAPCA = 6
# CLRAE = 7
# CODAAE = 8
# NONPARACODAPCA = 9
#
#
#
def coda_pca_run( reduce, X_train, X_validation, X_test, algorithm, learning_rate ):
    if algorithm == "CODA_PCA":
        alg = Alg.CODAPCA
    elif algorithm == "CODA_AE":
        alg = Alg.CODAAE


    if alg in [Alg.CODAPCA, Alg.SCODAPCA, Alg.CLRAE, Alg.CODAAE]:
        pca = CodaPCA(reduce,
                      lrate = learning_rate,
                      nn_shape = [100,100],
                      batchsize= 32,
                      alg=alg)

        pca.fit(X_train, epochs=300, repeat=1, verbose=True)
        return pca.transform(X_train), pca.transform(X_validation), pca.transform(X_test)

    else :
        pca = NonParametricCodaPCA(reduce)
        return pca.fit_transform(X_train), pca.transform(X_validation),  pca.transform(X_test)