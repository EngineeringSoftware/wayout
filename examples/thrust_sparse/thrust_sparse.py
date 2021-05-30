from kernels import *
from wayout.dynamic import Timer

import argparse

def print_sparse_vector(A_index, A_value):
    assert(A_index.size() == A_value.size())

    d_index = host_vector(int)(A_index)
    d_value = host_vector(float)(A_value)

    # for i in range(A_index.size()):
    #     print(f"({d_index[i]},{d_value[i]}) ", end='')
    print()


def sum_sparse_vectors(A_index, A_value, B_index, B_value, C_index, C_value):
    assert(A_index.size() == A_value.size())
    assert(B_index.size() == B_value.size())

    A_size = A_index.size()
    B_size = B_index.size()

    temp_index = device_vector(int)(A_size + B_size)
    temp_value = device_vector(float)(A_size + B_size)

    merge_by_key(A_index.begin(), A_index.end(),
            B_index.begin(), B_index.end(),
            A_value.begin(), B_value.begin(),
            temp_index.begin(), temp_value.begin())

    C_size = inner_product(temp_index.begin(), temp_index.end() - 1,
        temp_index.begin() - (-1), 
        0, 
        plus(int)(), 
        not_equal_to(int)()) + 1

    C_index.resize(C_size)
    C_value.resize(C_size)

    reduce_by_key(temp_index.begin(), temp_index.end(),
        temp_value.begin(), 
        C_index.begin(), C_value.begin(),
        equal_to(int)(),
        plus(float)())


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

    # A_index = device_vector(int)(4)
    # A_value = device_vector(float)(4)

    # # initialize sparse vector A with 6 elements
    # A_index[0] = 2;  A_value[0] = 10
    # A_index[1] = 3;  A_value[1] = 60
    # A_index[2] = 5;  A_value[2] = 20
    # A_index[3] = 8;  A_value[3] = 40

    # # initialize sparse vector B with 6 elements
    # B_index = device_vector(int)(6)
    # B_value = device_vector(float)(6)
    # B_index[0] = 1;  B_value[0] = 50
    # B_index[1] = 2;  B_value[1] = 30
    # B_index[2] = 4;  B_value[2] = 80
    # B_index[3] = 5;  B_value[3] = 30
    # B_index[4] = 7;  B_value[4] = 90
    # B_index[5] = 8;  B_value[5] = 10

    A_index = device_vector(int)(N)
    A_value = device_vector(float)(N)
    B_index = device_vector(int)(N)
    B_value = device_vector(float)(N)

    init_sparse_vector(A_index, A_value, B_index, B_value)

    # compute sparse vector C = A + B
    C_index = device_vector(int)();
    C_value = device_vector(float)();

    init_time = timer.seconds()
    timer.reset()

    for i in range(args.M):
        sum_sparse_vectors(A_index, A_value, B_index, B_value, C_index, C_value);

    print("Computing C = A + B for sparse vectors A and B");
    print("A ", end=''); print_sparse_vector(A_index, A_value);
    print("B ", end=''); print_sparse_vector(B_index, B_value);
    print("C ", end=''); print_sparse_vector(C_index, C_value);

    time = timer.seconds()

    # outupt timing info
    if args.file:
        with open(args.file, "a") as f:
            # workload, cuda, size1, size2, time
            f.write("%s, %d, %d, %d, %f, %f\n" % ("sparse", args.cuda, N, 1, init_time, time))


if __name__ == "__main__":
    run()
