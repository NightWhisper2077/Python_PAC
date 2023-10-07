import argparse
import random

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('input_frs')
parser.add_argument('input_scn')
parser.add_argument('chance', type=float)
args = parser.parse_args()


def read_arr(inp1, inp2):
    with open(args.input_frs) as f:
        str = f.readline()
        arr1 = str.split(' ')

        count = len(arr1)

    with open(args.input_scn) as f:
        str = f.readline()
        arr2 = str.split(' ')

    np_arr1 = np.array(arr1, dtype=int)
    np_arr2 = np.array(arr2, dtype=int)

    return np_arr1, np_arr2, count


def gen_index(arr, chn, cnt):
    if random.random() <= chn:
        return arr + cnt
    else:
        return arr


def variant_1():
    np_arr1, np_arr2, count = read_arr(args.input_frs, args.input_scn)
    result = np.concatenate((np_arr1, np_arr2))

    vfun = np.vectorize(gen_index)
    index = vfun(range(count), args.chance, count)

    """Result"""
    print(result[index])

    """Distribution of elements from the second array"""
    c = []

    for i in range(10000):
        vfun = np.vectorize(gen_index)
        index = vfun(range(count), args.chance, count)
        c.append(len(index[index >= count])/count)

    d = np.array(c)
    print(d.mean())
def variant_2():
    np_arr1, np_arr2, count = read_arr(args.input_frs, args.input_scn)

    res = np.random.choice((True, False), count, p=(args.chance, 1-args.chance))
    np_arr1[res] = np_arr2[res]

    """Result"""
    print(np_arr1)

    """Distribution of elements from the second array"""
    c = []

    for i in range(10000):
        res = np.random.choice((True, False), count, p=(args.chance, 1 - args.chance))
        c.append(len(res[res == True]) / count)

    d = np.array(c)
    print(d.mean())

def variant_3():
    np_arr1, np_arr2, count = read_arr(args.input_frs, args.input_scn)

    res = np.random.rand(count)
    np_arr1[res < args.chance] = np_arr2[res < args.chance]

    """Result"""
    print(np_arr1)

    """Distribution of elements from the second array"""
    c = []

    for i in range(10000):
        res = np.random.rand(count)
        c.append(len(res[res < args.chance]) / count)

    d = np.array(c)
    print(d.mean())

variant_1()
variant_2()
variant_3()
