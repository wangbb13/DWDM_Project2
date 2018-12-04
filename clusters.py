# -*- coding: utf-8 -*-
import os
import math
import random
from pca import decompose
from read_data import DataFrame
import matplotlib.pyplot as plt
import heapq
import numpy as np
from collections import deque


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


class KMeans(object):
    """docstring for KMeans"""
    def __init__(self, data, distance, k, iters=100):
        self.data = data
        self.distance = distance
        self.k = k
        self.data_size = len(data)
        self.cols = data.cols()
        self.iterations = iters

    def run(self):
        """
        :notation: 0,1,2..: label
        :process:
        repeat:
            assignment
            update
        until convergence
        :return: clustering label list
        """
        # initial part
        self.centroids = [self.gen() for _ in range(self.k)]
        self.groups = [-1 for _ in range(self.data_size)]
        self.flag = True
        # assignment
        def assignment():
            self.flag = True
            for i in range(self.data_size):
                assign = -1
                minval = 0xffffffff
                for j in range(self.k):
                    d = self.distance(self.centroids[j], self.data[i])
                    if d < minval:
                        minval = d
                        assign = j
                if self.groups[i] != assign:
                    self.flag = False
                self.groups[i] = assign
        # update centroids
        def update():
            numbers = [0 for _ in range(self.k)]
            sums = [[0 for _ in range(self.cols)] for _ in range(self.k)]
            for i in range(self.data_size):
                numbers[self.groups[i]] += 1
                sums[self.groups[i]] = Vector.add(sums[self.groups[i]], self.data[i])
            self.centroids = [[sums[_][j] / numbers[_] for j in range(self.cols)] 
                         if numbers[_] > 0 else self.centroids[_]
                         for _ in range(self.k)]
        # square error of all clusters
        def sse():
            ans = 0
            for i in range(self.data_size):
                ans += self.distance(self.data[i], self.centroids[self.groups[i]])
            return ans
        # iterate until convergence
        i = 0
        while i < self.iterations:
            i += 1
            print('Iteration', i, end=' ')
            assignment()
            update()
            print('SSE =', sse())
            if self.flag:
                break
        # return result
        return self.groups

    def gen(self):
        l, r = 0, self.data_size - 1
        return [self.data.col(_)[random.randint(l, r)] for _ in range(self.cols)]


class KMeansPCA(object):
    """docstring for KMeansPCA"""
    def __init__(self, data, distance, k, n, iters=100):
        _data_ = DataFrame(decompose(data.get_data(), n))
        self.kmeans = KMeans(_data_, distance, k, iters)
        self.dim = n

    def run(self):
        print('KMeans With PCA dimension =', self.dim)
        return self.kmeans.run()


class DBScan(object):
    """docstring for DBScan"""
    def __init__(self, data, distance, max_k=10):
        self.data = data
        self.distance = distance
        self.size = len(data)
        self.maxk = max_k
        self.eps = 2.25
        self.minpts = 5

    def __pre_processing__(self):
        """
        Find eps and minpts By Ploting
        """
        topk_d_file = 'topkd.txt'
        mat = [[0 for _ in range(self.size)] for _ in range(self.size)]
        if os.path.isfile(topk_d_file):
            with open(topk_d_file, 'r') as fin:
                sorted_mat = eval(fin.read().strip())
        else:
            for _ in range(self.size):
                for __ in range(_+1, self.size):
                    mat[_][__] = mat[__][_] = self.distance(self.data[_], self.data[__])
            sorted_mat = [sorted(heapq.nlargest(self.maxk, mat[_])) for _ in range(self.size)]
            with open(topk_d_file, 'w') as fout:
                fout.write(str(sorted_mat))
        def f1():
            for k in range(self.maxk, 1, -1):
                dy = sorted([sorted_mat[_][self.maxk-k] for _ in range(self.size)])
                ay = np.array(dy)
                ax = np.array([_ for _ in range(self.size)])
                plt.plot(ax, ay)
                plt.show()
                l = int(input('left  ='))
                r = int(input('right ='))
                print(dy[l:r])
        def f2():
            print('Total Points =', self.size)
            while True:
                try:
                    threshold = float(input('threshold = '))
                except Exception:
                    break
                for k in range(self.maxk, 1, -1):
                    dy = [sorted_mat[_][self.maxk-k] for _ in range(self.size)]
                    nu = sum([1 for _ in dy if _ <= threshold])
                    print('MinPTs =', k, 'Points =', nu)
        # f1()
        f2() # select eps=2.25 minpts=5

    def run(self):
        """
        :notation: 0: unvisited; 1,2,..: label; size+1: noise
        :process: directly (don't want to write, actually..)
        :return: clustering label list
        """
        # calc distance & neighbors
        neighbors = [[] for _ in range(self.size)]
        for _ in range(self.size):
            for __ in range(_+1, self.size):
                if self.distance(self.data[_], self.data[__]) <= self.eps:
                    neighbors[_].append(__)
                    neighbors[__].append(_)
        print('calc neighbors done.')
        # main part
        groups = [0 for _ in range(self.size)]
        noise_q = deque()
        label = 1
        noise = self.size + 1
        for _ in range(self.size):
            print(_)
            if groups[_] == 0:
                if len(neighbors[_]) < self.minpts:
                    flag = True
                    for p in neighbors[_]:
                        if len(neighbors[p]) >= self.minpts:
                            flag = False
                            break
                    if flag:
                        groups[_] = noise
                        noise_q.append(_)
                        print('noise point', _)
                        continue
                myq = deque([_])
                while len(myq):
                    p = myq.pop()
                    print(p, end=' ')
                    groups[p] = label
                    for np in neighbors[p]:
                        if groups[np] == 0:
                            myq.append(np)
                label += 1
        print('label = 1 ...', label)
        # tackle noise points
        print('noise points =', len(noise_q))
        while len(noise_q):
            rm = []
            for _ in noise_q:
                minval, select = 0xffffffff, noise
                for __ in neighbors[_]:
                    d = self.distance(self.data[_], self.data[__])
                    if d < minval and groups[__] < noise:
                        minval, select = d, groups[__]
                if select < noise:
                    groups[_] = select
                    rm.append(_)
            if len(rm):
                for _ in rm:
                    noise_q.remove(_)
            else:
                break
        print('noise points=', len(noise_q))
        # return the result
        return groups
