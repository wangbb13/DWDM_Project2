# -*- coding: utf-8 -*-
from read_data import Data
import numpy as np 
from sklearn.decomposition import PCA


def decompose(data, n):
    pca = PCA(n_components=n)
    pca.fit(data)
    x_pca = pca.transform(data)
    return x_pca.tolist()
