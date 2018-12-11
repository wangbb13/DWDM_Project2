# -*- coding: utf-8 -*-
import sys
from kits import Data, Euclidean, Cosine
from clusters import KMeans, KMeansPCA, DBScan, DBScanPCA
from evaluate import Purity, FScore


def main():
    filename = 'Frogs_MFCCs.csv'
    frog = Data(filename)
    d = Euclidean.distance
    classes = frog.get_gt()
    k_gt = 4
    print('k-means')
    k_c = 4
    kmeans_pca = KMeansPCA(frog, d, k_c, 10)
    p, f = 0, 0
    for _ in range(10):
        cluster = kmeans_pca.run()
        p_ = Purity(cluster, classes, k_c, k_gt)
        f_ = FScore(cluster, classes)
        if f_ > f:
            p, f = p_, f_
    print('purity =', p)
    print('f-score =', f)
    print('========')
    print('dbscan')
    dbscan_pca = DBScanPCA(frog, d, 10)
    cluster, k_c = dbscan_pca.run()
    p = Purity(cluster, classes, k_c, k_gt)
    f = FScore(cluster, classes)
    print('purity =', p)
    print('f-score =', f)
    print('========')


if __name__ == '__main__':
    main()
