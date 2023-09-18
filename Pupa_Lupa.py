def matrix_scan(f):
    """Function reads the lines of the file f up to the separator spr,
    then returns the dimension and the matrix itself"""

    with open(f) as file:
        row = 0
        column = 0
        matrix = []

        while True:
            str = file.readline()
            if str == '':
                break
            else:
                row += 1

            numbers = str.split(' ')
            column = len(numbers)
            matrix.append(numbers)

        return row, column, matrix

def matrix_summ(sym, mtr1, mtr2, row, column):
    """This function sums or subtracts two matrices of the same size
    by elements depending on the sign"""

    res = []

    for i in range(row):
        str = []
        for j in range(column):
            try:
                if sym == '+':
                    str.append(int(mtr1[i][j]) + int(mtr2[i][j]))
                else:
                    str.append(int(mtr1[i][j]) - int(mtr2[i][j]))
            except ValueError:
                print("Incorrect matrix data type")
                exit()

        res.append(str)

    return res

class Accountant():
    """The class through which the Pupa and Lupa classes are called.
    The take_salary() function calls the increment method of the worker instance to increment by count.
    The call_worker function calls the sum/difference method of the worker instance"""

    def __init__(self, filename1, filename2):
        self._file1 = filename1
        self._file2 = filename2
        pass
    def give_salary(self, worker, count):
        worker.take_salary(count)

    def call_worker(self, worker):
        worker.do_work(self._file1, self._file2)

class Pupa():
    """The Pupa class is needed for summing matrices from files and incrementing the counter"""
    def __init__(self, inc = 0):
        self._inc = inc
    def take_salary(self, inc):
        self._inc += inc

    def do_work(self, filename1, filename2):
        cnt_row1, cnt_column1, matrix1 = matrix_scan(filename1)
        cnt_row2, cnt_column2, matrix2 = matrix_scan(filename2)

        if (cnt_row1 == cnt_row2 and cnt_column1 == cnt_column2):
            res = matrix_summ('+', matrix1, matrix2, cnt_row1, cnt_column1)
        else:
            print("Incorrect dimension of matrices")
            exit()

        for i in res:
            print(i)

class Lupa():
    """The Lupa class is needed for subtraction matrices from files and incrementing the counter"""
    def __init__(self, inc = 0):
        self._inc = inc

    def take_salary(self, inc):
        self._inc += inc

    def do_work(self, filename1, filename2):
        cnt_row1, cnt_column1, matrix1 = matrix_scan(filename1)
        cnt_row2, cnt_column2, matrix2 = matrix_scan(filename2)

        if (cnt_row1 == cnt_row2 and cnt_column1 == cnt_column2):
            res = matrix_summ('-', matrix1, matrix2, cnt_row1, cnt_column1)
        else:
            print("Incorrect dimension of matrices")
            exit()

        for i in res:
            print(i)