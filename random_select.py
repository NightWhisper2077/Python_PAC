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
    if random.random() < chn:
        return arr + cnt
    else:
        return arr


def gen_index2(arr, chn):
    if random.random() < chn:
        return 0
    else:
        return 1


def gen_inv(arr):
    if arr == 1:
        return 0
    else:
        return 1


def variant_1():
    np_arr1, np_arr2, count = read_arr(args.input_frs, args.input_scn)
    result = np.concatenate((np_arr1, np_arr2))

    vfun = np.vectorize(gen_index)
    index = vfun(range(count), args.chance, count)
    print(result[index])


def variant_2():
    np_arr1, np_arr2, count = read_arr(args.input_frs, args.input_scn)

    vfun = np.vectorize(gen_index2)
    rnd_arr = vfun(range(count), args.chance)
    vfun = np.vectorize(gen_inv)

    print(np_arr1 * rnd_arr + np_arr2 * vfun(rnd_arr))


variant_1()

variant_2()
