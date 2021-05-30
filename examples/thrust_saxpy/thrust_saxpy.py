from kernels import *
from wayout.dynamic import Timer

import argparse

def saxpy_slow(A, X, Y):
    temp = device_vector(float)(X.size())

    fill(temp.begin(), temp.end(), A)

    transform(X.begin(), X.end(), temp.begin(), temp.begin(), multiplies(float)())

    transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), plus(float)())

def run() -> None: 
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, 
            help="determines number of rows (columns) (default: 2^10 = 1024)")
    parser.add_argument('-M', type=int, default=1, help="iteration")
    parser.add_argument('--cuda', action="store_true", help="use CUDA (default: 0)")
    parser.add_argument('--file', type=str, help="output timing info to file")

    args = parser.parse_args()
    N = 4
    if args.N:
        N = args.N
    timer = Timer()

    # x = host_vector(float)(4)
    # y = host_vector(float)(4)
    # for i in range(4):
    #     x[i] = 1
    #     y[i] = i + 1

    x = host_vector(float)(N)
    y = host_vector(float)(N)
    init_saxpy_vector(x, y)
    X = device_vector(float)(x)
    Y = device_vector(float)(y)

    init_time = timer.seconds()
    timer.reset()

    for i in range(args.M):
        # X = device_vector(float)(x)
        # Y = device_vector(float)(y)
        saxpy_slow(2.0, X, Y)
       
        # saxpy_fast not implemented because functor required

    # copy(Y.begin(), Y.end(), y.begin())
    # for i in range(4):
    #     print(y[i], end=', ')
    # print()

    time = timer.seconds()

    # outupt timing info
    if args.file:
        with open(args.file, "a") as f:
            # workload, cuda, size1, size2, time
            f.write("%s, %d, %d, %d, %f, %f\n" % ("saxpy", args.cuda, N, 1, init_time, time))


if __name__ == "__main__":
    run()
