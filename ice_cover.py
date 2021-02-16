import numpy as np
import math
import random


def get_dataset():
    years = []
    days = []
    ans = []
    file = open('ice_history.txt')
    for line in file:
        field = line.split('-')

        if field[0] != '\n':
            if int(field[0]) > 1000:
                years.append(int(field[0]))
            else:
                days.append(int(field[0]))
    for i in range(0, len(years)):
        ans.append([years[i], days[i]])
    return ans


# returns separated numpy array for easier calculations
def get_arr(n):
    dataset = get_dataset()
    arrx = np.empty(0)
    for item in dataset:
        arrx = np.append(arrx, item[0])
    arry = np.empty(0)
    for item in dataset:
        arry = np.append(arry, item[1])
    if n == 1:
        return arry

    return arry, arrx


def print_stats(dataset):
    arr = get_arr(1)

    print(len(arr), round(np.mean(arr), 2), round(np.std(arr), 2), sep='\n')


def regression(beta_0, beta_1):
    arry, arrx = get_arr(2)
    mse = 0
    for i in range(len(arrx)):
        mse += (beta_0 + (beta_1 * arrx[i]) - arry[i]) ** 2

    return mse / len(arrx)


def gradient_descent(beta_0, beta_1):
    arry, arrx = get_arr(2)
    mse0 = 0
    mse1 = 0
    for i in range(len(arrx)):
        mse0 += (beta_0 + (beta_1 * arrx[i]) - arry[i])
        mse1 += (beta_0 + (beta_1 * arrx[i]) - arry[i]) * arrx[i]
    mse0 = (mse0 / len(arrx)) * 2
    mse1 = (mse1 / len(arrx)) * 2
    return mse0, mse1


def iterate_gradient(T, eta):
    beta_0, beta_1 = 0, 0

    for i in range(T):
        gd0, gd1 = gradient_descent(beta_0, beta_1)
        beta_0 = beta_0 - (eta * gd0)
        beta_1 = beta_1 - (eta * gd1)
        print(i + 1, round(beta_0, 2), round(beta_1, 2), round(regression(beta_0, beta_1), 2))


def compute_betas():
    arry, arrx = get_arr(2)
    xmean = np.mean(arrx)
    ymean = np.mean(arry)
    xsum = 0
    xsq = 0
    for i in range(len(arrx)):
        xsum += (arrx[i] - xmean) * (arry[i] - ymean)

        xsq += (arrx[i] - xmean) ** 2
    b_one = xsum / xsq
    b_zero = ymean - (b_one * xmean)
    mse = 0

    for i in range(len(arrx)):
        mse += (b_zero + (b_one * arrx[i]) - arry[i]) ** 2

    mse = mse / len(arrx)
    return b_zero, b_one, round(mse, 2)


def predict(year):
    b_zero, b_one, mse = compute_betas()
    return round(b_zero + (b_one * year), 2)


# normalized functions for  ngd
def gradient_d_norm(beta_0, beta_1):
    arry, arrx = get_arr(2)
    xmean = np.mean(arrx)

    stdx = 0
    n = len(arrx)
    for i in range(n):
        stdx += (arrx[i] - xmean) ** 2

    stdx = math.sqrt(stdx / (n - 1))

    for i in range(n):
        arrx[i] = (arrx[i] - xmean) / stdx

    mse0 = 0
    mse1 = 0
    for i in range(len(arrx)):
        mse0 += (beta_0 + (beta_1 * arrx[i]) - arry[i])
        mse1 += (beta_0 + (beta_1 * arrx[i]) - arry[i]) * arrx[i]
    mse0 = (mse0 / len(arrx)) * 2
    mse1 = (mse1 / len(arrx)) * 2
    return mse0, mse1


def regression_norm(beta_0, beta_1):
    arry, arrx = get_arr(2)
    xmean = np.mean(arrx)

    stdx = 0
    n = len(arrx)
    for i in range(n):
        stdx += (arrx[i] - xmean) ** 2

    stdx = math.sqrt(stdx / (n - 1))

    for i in range(n):
        arrx[i] = (arrx[i] - xmean) / stdx
    mse = 0
    for i in range(len(arrx)):
        mse += (beta_0 + (beta_1 * arrx[i]) - arry[i]) ** 2

    return round(mse / len(arrx), 2)


def iterate_normalized(T, eta):
    beta_0, beta_1 = 0, 0

    for i in range(T):
        gd0, gd1 = gradient_d_norm(beta_0, beta_1)
        beta_0 = beta_0 - (eta * gd0)
        beta_1 = beta_1 - (eta * gd1)
        print(i + 1, round(beta_0, 2), round(beta_1, 2), regression_norm(beta_0, beta_1))


# rand functions for sgd
def gradient_rand(beta_0, beta_1, ran):
    arry, arrx = get_arr(2)
    xmean = np.mean(arrx)

    stdx = 0
    n = len(arrx)
    for i in range(n):
        stdx += (arrx[i] - xmean) ** 2

    stdx = math.sqrt(stdx / (n - 1))

    for i in range(n):
        arrx[i] = (arrx[i] - xmean) / stdx

    mse0 = (beta_0 + (beta_1 * arrx[ran]) - arry[ran])
    mse1 = (beta_0 + (beta_1 * arrx[ran]) - arry[ran]) * arrx[ran]

    mse0 = mse0 * 2
    mse1 = mse1 * 2
    return mse0, mse1


def regression_rand(beta_0, beta_1, ran):
    arry, arrx = get_arr(2)
    xmean = np.mean(arrx)

    stdx = 0
    n = len(arrx)
    for i in range(n):
        stdx += (arrx[i] - xmean) ** 2

    stdx = math.sqrt(stdx / (n - 1))

    for i in range(n):
        arrx[i] = (arrx[i] - xmean) / stdx

    mse = (beta_0 + (beta_1 * arrx[ran]) - arry[ran]) ** 2

    return round(mse, 2)


def sgd(T, eta):
    beta_0, beta_1 = 0, 0

    for i in range(T):
        ran = random.randrange(0, 165, 1)
        gd0, gd1 = gradient_rand(beta_0, beta_1, ran)
        beta_0 = beta_0 - (eta * gd0)
        beta_1 = beta_1 - (eta * gd1)
        print(i + 1, round(beta_0, 2), round(beta_1, 2), regression_rand(beta_0, beta_1, ran))


if __name__ == "__main__":
    data = get_dataset()
    print('data: ', data)
    print_stats(data)
    print('regression: ', regression(0, 0))
    print('gradient_descent: ', gradient_descent(0, 0))
    iterate_gradient(5, 1e-7)
    print('compute_betas: ', compute_betas())
    print('predict: ', predict(2021))
    iterate_normalized(5, 0.1)
    sgd(5, 0.1)
