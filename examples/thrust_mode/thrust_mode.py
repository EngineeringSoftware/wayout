from kernels import *

import argparse

def run() -> None: 
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, 
            help="determines number of rows (columns) (default: 2^10 = 1024)")
    parser.add_argument('-M', type=int, default=1, help="iteration")
    parser.add_argument('--cuda', action="store_true", help="use CUDA (default: 0)")
    parser.add_argument('--file', type=str, help="output timing info to file")
    
    args = parser.parse_args()
    N = 30
    if args.N:
        N = args.N
    timer = pk.Timer()

    M = 10
    # rng = linear_congruential_engine(int, 48271, 0, 2147483647)() # typedef to default_random_engine
    # dist = uniform_int_distribution(int)(0, M-1)

    # # generate random data on the host
    # h_data = host_vector(int)(N)
    # for i in range(N):
    #     h_data[i] = dist(rng)
    h_data = host_vector(int)(N)
    init_mode_vector(h_data, M)

    # transfer data to device
    d_data = device_vector(int)(h_data)

    init_time = timer.seconds()
    timer.reset()
    
    for i in range(args.M):
        # print the initial data
        print("initial data")
        # copy(d_data.begin(), d_data.end(), h_data.begin())
        # for i in range(N):
        #     print(h_data[i], end=' ')
        print()

        # sort data to bring equal elements together
        sort(d_data.begin(), d_data.end())
        
        # print the sorted data
        print("sorted data")
        # copy(d_data.begin(), d_data.end(), h_data.begin())
        # for i in range(N):
        #     print(h_data[i], end=' ')
        print()

        # count number of unique keys
        num_unique = inner_product(d_data.begin(), d_data.end() - 1,
            d_data.begin() - (-1),
            0,
            plus(int)(),
            not_equal_to(int)()) + 1

        # count multiplicity of each key
        d_output_keys = device_vector(int)(num_unique)
        d_output_counts = device_vector(int)(num_unique)
        reduce_by_key(d_data.begin(), d_data.end(),
                              constant_iterator(int)(1),
                              d_output_keys.begin(),
                              d_output_counts.begin())
        
        # print the counts
        h_key = host_vector(int)(d_output_keys)
        h_count = host_vector(int)(d_output_counts)
        print("values")
        # for i in range(num_unique):
        #     print(h_key[i], end=' ')
        print()

        # print the values 
        print("counts")
        # for i in range(num_unique):
        #     print(h_count[i], end=' ')
        print()

        # find the index of the maximum count
        # mode_iter = max_element(d_output_counts.begin(), d_output_counts.end())
        mode_iter = max_element(h_count.begin(), h_count.end())

        mode = h_key[mode_iter - h_count.begin()]
        occurances = mode_iter.__deref__()
        
        print(f"Modal value {mode} occurs {occurances} times ")

    time = timer.seconds()

    # outupt timing info
    if args.file:
        with open(args.file, "a") as f:
            # workload, cuda, size1, size2, time
            f.write("%s, %d, %d, %d, %f, %f\n" % ("mode", args.cuda, N, 1, init_time, time))


if __name__ == "__main__":
    run()
