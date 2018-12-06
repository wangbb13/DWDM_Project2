# -*- coding: utf-8 -*-
from kits import Data, Euclidean, Cosine
from clusters import KMeans, KMeansPCA, DBScan, DBScanPCA
from evaluate import Purity, FScore
import matplotlib.pyplot as plt
import numpy as np


def test_case_1():
    print()
    filename = 'Frogs_MFCCs.csv'
    frog = Data(filename)
    print('len(frog) =', len(frog))
    print('frog[0] =', frog[0])
    print('label of frog[0] =', frog.get_f_label(0))
    print('euclidean between [0] and [1] =', Euclidean.distance(frog[0], frog[1]))


def test_case_2():
    print()
    vec_a = [0, 1]
    vec_b = [1, 0]
    distance = Euclidean.distance
    print('vec_a =', vec_a)
    print('vec_b =', vec_b)
    print('euclidean =', distance(vec_a, vec_b))
    distance = Cosine.distance
    print('cosine =', distance(vec_a, vec_b))


def test_case_3():
    filename = 'Frogs_MFCCs.csv'
    frog = Data(filename)
    d = Euclidean.distance
    k_c, k_gt = 4, 4
    kmeans = KMeans(frog, d, k_c)
    classes = frog.get_gt()
    cluster = kmeans.run()
    print('purity =', Purity(cluster, classes, k_c, k_gt))
    print('f-score =', FScore(cluster, classes))


def test_case_4():
    filename = 'Frogs_MFCCs.csv'
    frog = Data(filename)
    d = Euclidean.distance
    k_c, k_gt = 4, 4
    x = list(range(1, 23))
    y1, y2 = [], []
    for dim in range(1, 23):
        kmeans_pca = KMeansPCA(frog, d, k_c, dim)
        classes = frog.get_gt()
        p, f = 0, 0
        for _ in range(10):
            cluster = kmeans_pca.run()
            p_ = Purity(cluster, classes, k_c, k_gt)
            f_ = FScore(cluster, classes)
            if p_ > p and f_ > p:
                p, f = p_, f_
        y1.append(p)
        y2.append(f)
        print('purity =', p)
        print('f-score =', f)
    x = np.array(x)
    y1 = np.array(y1)
    y2 = np.array(y2)
    plt.plot(x, y1, 'g-', marker='o', label='purity', linewidth=2)
    plt.plot(x, y2, 'r-', marker='*', label='f-score', linewidth=2)
    plt.xticks(x)
    plt.legend(loc=0)
    plt.show()


def test_case_5():
    filename = 'Frogs_MFCCs.csv'
    frog = Data(filename)
    d = Euclidean.distance
    k_c, k_gt, dim = 4, 4, 11
    N = 30
    x = list(range(1, N+1))
    y1, y2 = [], []
    kmeans_pca = KMeansPCA(frog, d, k_c, dim)
    classes = frog.get_gt()
    p_, f_ = 0, 0
    for _ in range(N):
        cluster = kmeans_pca.run()
        p = Purity(cluster, classes, k_c, k_gt)
        f = FScore(cluster, classes)
        y1.append(p)
        y2.append(f)
        if p > p_ and f > f_:
            p_, f_ = p, f
        print('purity =', p)
        print('f-score =', f)
    print('optional purity =', p_)
    print('optional f-score =', f_)
    x = np.array(x)
    y1 = np.array(y1)
    y2 = np.array(y2)
    plt.plot(x, y1, 'g-', marker='o', label='purity', linewidth=2)
    plt.plot(x, y2, 'r-', marker='*', label='f-score', linewidth=2)
    plt.xticks(x)
    plt.legend(loc=0)
    plt.show()


def test_case_6():
    filename = 'Frogs_MFCCs.csv'
    frog = Data(filename)
    d = Euclidean.distance
    dbscan = DBScan(frog, d)
    dbscan.__pre_processing__()


def test_case_7():
    filename = 'Frogs_MFCCs.csv'
    frog = Data(filename)
    d = Euclidean.distance
    dbscan = DBScan(frog, d)
    classes = frog.get_gt()
    k_gt = 4
    cluster, k_c = dbscan.run()
    print('purity =', Purity(cluster, classes, k_c, k_gt))
    print('f-score =', FScore(cluster, classes))


def test_case_8():
    filename = 'Frogs_MFCCs.csv'
    frog = Data(filename)
    d = Euclidean.distance
    n = 9
    classes = frog.get_gt()
    k_gt = 4
    pl = []
    fl = []
    for mp in range(4, 20, 2):
        dbscan_pca = DBScanPCA(frog, d, n, eps=1.75, minpts=mp)
        cluster, k_c = dbscan_pca.run()
        p = Purity(cluster, classes, k_c, k_gt)
        f = FScore(cluster, classes)
        print('purity =', p)
        print('f-score =', f)
        pl.append(p)
        fl.append(f)
    ax = np.array([_ for _ in range(4, 20, 2)])
    ap = np.array(pl)
    af = np.array(fl)
    plt.plot(ax, ap, 'g-', marker='o', label='purity', linewidth=2)
    plt.plot(ax, af, 'r-', marker='*', label='f-score', linewidth=2)
    plt.xticks(ax)
    plt.legend(loc=0)
    plt.show()


def test_case_9():
    filename = 'Frogs_MFCCs.csv'
    frog = Data(filename)
    d = Euclidean.distance
    n = 9
    dbscan_pca = DBScanPCA(frog, d, n)
    dbscan_pca.__pre_processing__()


def main():
    # test_case_1()
    # test_case_2()
    # test_case_3()
    # test_case_4()
    # test_case_5()
    # test_case_6()
    # test_case_7()
    test_case_8()
    # test_case_9()


if __name__ == '__main__':
    main()
