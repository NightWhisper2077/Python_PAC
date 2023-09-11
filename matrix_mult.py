import math
import random
import os
import argparse

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()

def matrix_scan(f, spr):
    """Function reads the lines of the file f up to the separator spr,
    then returns the dimension and the matrix itself"""
    row = 0
    column = 0
    matrix = []

    while True:
        str = f.readline()
        if str == spr:
            break
        else:
            row += 1

        numbers = str.split(' ')
        column = len(numbers)
        matrix.append(numbers)

    return row, column, matrix

def matrix_mult(f):
    """The function multiplies two matrices by means of a triple loop,
    returns the result and the number of rows to output"""
    row_1, column_1, matrix_1 = matrix_scan(f, '\n')
    row_2, column_2, matrix_2 = matrix_scan(f, '')

    res_matrix = []
    if (column_1 == row_2):
        for i in range(row_1):
            lst = []
            for k in range(column_2):
                a = 0
                for m in range(column_1):
                    try:
                        a += int(matrix_1[i][m]) * int(matrix_2[m][k])
                    except ValueError:
                        print("Incorrect matrix data type")
                        exit()
                lst.append(a)
            res_matrix.append(lst)
    else:
        print("Incorrect dimension of matrices")
        exit()

    return res_matrix, row_1

def main():
    with open(args.input) as f:
        res, count = matrix_mult(f)

        for i in range(count):
            print(res[i])

main()