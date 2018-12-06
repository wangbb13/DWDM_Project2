# -*- coding: utf-8 -*-
import os
import csv
import math
from sklearn.decomposition import PCA


class DataFrame(object):
    """docstring for DataFrame"""
    def __init__(self, twodlist):
        self.data = twodlist
        self._size_ = len(self.data)
        self._col_ = len(self.data[0]) if self._size_ > 0 else 0
        self.column = [[self.data[_][c] for _ in range(self._size_)] for c in range(self._col_)]

    def __len__(self):
        return self._size_

    def __getitem__(self, index):
        return self.data[index]

    def cols(self):
        return self._col_

    def col(self, index):
        return self.column[index]


class Data(object):
    """docstring for Data"""
    def __init__(self, filename):
        if not os.path.exists(filename):
            print(filename, 'does not exist.')
            exit(0)
        self.filename = filename
        self.process()

    def process(self):
        map_dict = dict()
        fid, gid, sid = 0, 0, 0     # family, genus, species
        with open(self.filename, 'r') as fin:
            reader = csv.reader(fin)
            raw = [_ for _ in reader]
        self.header = raw[0]
        for item in raw[1:]:
            if item[22] not in map_dict:
                map_dict[item[22]] = fid
                fid += 1
            if item[23] not in map_dict:
                map_dict[item[23]] = gid
                gid += 1
            if item[24] not in map_dict:
                map_dict[item[24]] = sid
                sid += 1
        self.refine = [[float(_) for _ in item[:22]] + [map_dict[_] for _ in item[22:-1]] for item in raw[1:]]
        self._size_ = len(self.refine)
        self._col_ = 22
        self.column = [[self.refine[_][c] for _ in range(self._size_)] \
                       for c in range(self._col_)]
        self.gt = [self.refine[_][22] for _ in range(self._size_)]

    def __len__(self):
        return self._size_

    def __getitem__(self, index):
        return self.refine[index][:22]

    def get_f_label(self, index):
        return self.refine[index][22]

    def get_gt(self):
        return self.gt

    def cols(self):
        return self._col_

    def col(self, index):
        return self.column[index]

    def get_data(self):
        return self.refine


class Vector(object):
    """docstring for Vector"""
    @staticmethod
    def norm(vec):
        return math.sqrt(sum([_ ** 2 for _ in vec]))

    @staticmethod
    def dot(vec_a, vec_b):
        return sum([vec_a[_] * vec_b[_] for _ in range(len(vec_a))])

    @staticmethod
    def add(vec_a, vec_b):
        return [vec_a[_] + vec_b[_] for _ in range(len(vec_a))]

    @staticmethod
    def minus(vec_a, vec_b):
        return [vec_a[_] - vec_b[_] for _ in range(len(vec_a))]


class Euclidean(object):
    """docstring for EuclideanD"""
    @staticmethod
    def distance(vec_a, vec_b):
        return Vector.norm(Vector.minus(vec_a, vec_b))


class Cosine(object):
    """docstring for Cosine"""
    @staticmethod
    def distance(vec_a, vec_b):
        return Vector.dot(vec_a, vec_b) / (Vector.norm(vec_a) * Vector.norm(vec_b))


def decompose(data, n):
    pca = PCA(n_components=n)
    pca.fit(data)
    x_pca = pca.transform(data)
    return x_pca.tolist()
