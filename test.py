# -*- coding: utf-8 -*-
from read_data import Data
from clusters import Euclidean, Cosine, KMeans
from evaluate import Purity, FScore


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


def main():
    # test_case_1()
    # test_case_2()
    test_case_3()


if __name__ == '__main__':
    main()
