import argparse
import random

from kernels import *
from wayout.dynamic import Timer

def run() -> None: 
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, 
            help="determines number of rows (columns) (default: 2^10 = 1024)")
    parser.add_argument('-M', type=int, default=1, help="iteration")
    parser.add_argument('--cuda', action="store_true", help="use CUDA (default: 0)")
    parser.add_argument('--file', type=str, help="output timing info to file")

    args = parser.parse_args()
    N = 32 << 2
    if args.N:
        N = args.N
    timer = Timer()

    # h_vec = host_vector(int)(32 << 2)
    # generate(h_vec.begin(), h_vec.end(), rand)
    h_vec = host_vector(int)(N)
    init_sort_vector(h_vec)

    # print("initial")
    # for i in range(h_vec.size()):
    #     h_vec[i] = random.randint(0, 100)
    #     print(h_vec[i], end=',')
    # print()

    d_vec = device_vector(int)(h_vec)

    init_time = timer.seconds()
    timer.reset()

    for i in range(args.M):
        sort(d_vec.begin(), d_vec.end())

    copy(d_vec.begin(), d_vec.end(), h_vec.begin())

    # print("sorted")
    # for i in range(h_vec.size()):
    #     print(h_vec[i], end=',')
    # print()

    time = timer.seconds()

    # outupt timing info
    if args.file:
        with open(args.file, "a") as f:
            # workload, cuda, size1, size2, time
            f.write("%s, %d, %d, %d, %f, %f\n" % ("sort", args.cuda, N, 1, init_time, time))


if __name__ == "__main__":
    run()
