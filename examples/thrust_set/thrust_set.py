from kernels import *
from wayout.dynamic import Timer

import argparse

Vector = device_vector(int)

# helper routine
def print_set(s, v):
    print(f"{s} [", end='')
    # h_vec = host_vector(int)(v)
    # for i in range(v.size()):
    #     print(f" {h_vec[i]}", end='')
    print(" ]")


def Merge(A, B):
    # merged output is always exactly A.size() + B.size()
    C = Vector(A.size() + B.size())

    merge(A.begin(), A.end(), B.begin(), B.end(), C.begin())

    print_set("Merge(A,B)", C)


def SetUnion(A, B):
    # union output is at most A.size() + B.size()
    C = Vector(A.size() + B.size())

    # set_union returns an iterator C_end denoting the end of input

    C_end = set_union(A.begin(), A.end(), B.begin(), B.end(), C.begin())

    # shrink C to exactly fit output
    C.erase(C_end, C.end())

    print_set("Union(A,B)", C)


def SetIntersection(A, B):
    # intersection output is at most min(A.size(), B.size())
    C = Vector(min(A.size(), B.size()))

    # set_union returns an iterator C_end denoting the end of input
    C_end = set_intersection(A.begin(), A.end(), B.begin(), B.end(), C.begin())

    # shrink C to exactly fit output
    C.erase(C_end, C.end())

    print_set("Intersection(A,B)", C)


def SetDifference(A, B):
    # difference output is at most A.size()
    C = Vector(A.size())

    # set_union returns an iterator C_end denoting the end of input
    C_end = set_difference(A.begin(), A.end(), B.begin(), B.end(), C.begin())
    
    # shrink C to exactly fit output
    C.erase(C_end, C.end())

    print_set("Difference(A,B)", C)


def SetSymmetricDifference(A, B):
    # symmetric difference output is at most A.size() + B.size()
    C = Vector(A.size() + B.size())

    # set_union returns an iterator C_end denoting the end of input
    C_end = set_symmetric_difference(A.begin(), A.end(), B.begin(), B.end(), C.begin())
    
    # shrink C to exactly fit output
    C.erase(C_end, C.end())

    print_set("SymmetricDifference(A,B)", C)


def SetIntersectionSize(A, B):
    # computes the exact size of the intersection without allocating output
    C_begin = discard_iterator([])()
    C_end = set_intersection(A.begin(), A.end(), B.begin(), B.end(), C_begin)

    print(f"SetIntersectionSize(A,B) {C_end - C_begin}")


def run() -> None: 
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, 
            help="determines number of rows (columns) (default: 2^10 = 1024)")
    parser.add_argument('-M', type=int, default=1, help="iteration")
    parser.add_argument('--cuda', action="store_true", help="use CUDA (default: 0)")
    parser.add_argument('--file', type=str, help="output timing info to file")
    
    args = parser.parse_args()
    N = 7
    if args.N:
        N = args.N
    timer = Timer()

    # a = [0,2,4,5,6,8,9]
    # b = [0,1,2,3,5,7,8]

    # A = device_vector(int)(len(a))
    # B = device_vector(int)(len(b))

    # for i in range(len(a)):
    #     A[i] = a[i]
    # for i in range(len(b)):
    #     B[i] = b[i]

    A = device_vector(int)(N)
    B = device_vector(int)(N)
    init_set_vector(A, B)

    print_set("Set A", A)
    print_set("Set B", B)

    init_time = timer.seconds()
    timer.reset()

    for i in range(args.M):
        Merge(A,B)
        SetUnion(A,B)
        SetIntersection(A,B)
        SetDifference(A,B)
        SetSymmetricDifference(A,B)

        SetIntersectionSize(A,B)

    time = timer.seconds()

    # outupt timing info
    if args.file:
        with open(args.file, "a") as f:
            # workload, cuda, size1, size2, time
            f.write("%s, %d, %d, %d, %f, %f\n" % ("set", args.cuda, N, 1, init_time, time))


if __name__ == "__main__":
    run()
