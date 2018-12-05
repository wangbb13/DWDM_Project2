# -*- coding: utf-8 -*-

def Purity(vec_c, vec_gt, k_c, k_gt):
    n = len(vec_c)
    mat = [[0 for _ in range(k_gt)] for _ in range(k_c)]
    for i in range(n):
        if vec_c[i] > k_c:
            continue
        mat[vec_c[i]][vec_gt[i]] += 1
    return sum([max(_) for _ in mat]) / n


def FScore(vec_c, vec_gt):
    n = len(vec_c)
    tp, fp, fn = 0, 0, 0
    for i in range(n):
        for j in range(i+1, n):
            bc = vec_c[i] == vec_c[j]
            bgt = vec_gt[i] == vec_gt[j]
            if bc and bgt:
                tp += 1
            elif bc and not bgt:
                fp += 1
            elif not bc and bgt:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return (2 * precision * recall) / (precision + recall)
