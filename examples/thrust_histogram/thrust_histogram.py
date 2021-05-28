from kernels import *

import argparse

# simple routine to print contents of a vector
def print_vector(name, v):
    h_vec = host_vector(int)(v)
    print("  %20s  " % (name), end='')
    for i in range(v.size()):
        print(f" {h_vec[i]}", end='')
    print()
  

# dense histogram using binary search
def dense_histogram(input_, histogram):
    ValueType = int # input value type
    IndexType = int # histogram index type
  
    # copy input data (could be skipped if input is allowed to be modified)
    data = device_vector(ValueType)(input_)
      
    # print the initial data
    # print_vector("initial data", data)
  
    # sort data to bring equal elements together
    sort(data.begin(), data.end())
    
    # print the sorted data
    # print_vector("sorted data", data)
  
    # number of histogram bins is equal to the maximum value plus one
    # h_data = host_vector(ValueType)(data)
    # num_bins = h_data.back() + 1
    h_data = host_vector(ValueType)(1)
    copy(data.end()-1, data.end(), h_data.begin())
    num_bins = h_data[0] + 1
  
    # resize histogram storage
    histogram.resize(num_bins)
    
    # find the end of each bin of values
    search_begin = counting_iterator(IndexType)(0)
    upper_bound(data.begin(), data.end(),
                search_begin, search_begin - (-num_bins),
                histogram.begin())
    
    # print the cumulative histogram
    print_vector("cumulative histogram", histogram)
  
    # compute the histogram by taking differences of the cumulative histogram
    adjacent_difference(histogram.begin(), histogram.end(),
                                histogram.begin())
  
    # print the histogram
    print_vector("histogram", histogram)
  

# sparse histogram using reduce_by_key
def sparse_histogram(input_, histogram_values, histogram_counts):
    ValueType = int # input value type
    IndexType = int # histogram index type
  
    # copy input data (could be skipped if input is allowed to be modified)
    data = device_vector(ValueType)(input_)
      
    # print the initial data
    # print_vector("initial data", data)
  
    # sort data to bring equal elements together
    sort(data.begin(), data.end())
    
    # print the sorted data
    # print_vector("sorted data", data)
  
    # number of histogram bins is equal to number of unique values (assumes data.size() > 0)
    num_bins = inner_product(data.begin(), data.end() - 1,
                             data.begin() - (-1),
                             IndexType(1),
                             plus(IndexType)(),
                             not_equal_to(ValueType)())
  
    # resize histogram storage
    histogram_values.resize(num_bins)
    histogram_counts.resize(num_bins)
    
    # compact find the end of each bin of values
    reduce_by_key(data.begin(), data.end(),
                  constant_iterator(IndexType)(1),
                  histogram_values.begin(),
                  histogram_counts.begin())
    
    # print the sparse histogram
    print_vector("histogram values", histogram_values)
    print_vector("histogram counts", histogram_counts)
  
  
def run() -> None: 
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, 
            help="determines number of rows (columns) (default: 2^10 = 1024)")
    parser.add_argument('-M', type=int, default=1, help="iteration")
    parser.add_argument('--cuda', action="store_true", help="use CUDA (default: 0)")
    parser.add_argument('--file', type=str, help="output timing info to file")

    rng = linear_congruential_engine(int, 48271, 0, 2147483647)() # typedef to default_random_engine
    dist = uniform_int_distribution(int)(0, 9)
  
    N = 40
    S = 4

    args = parser.parse_args()
    if args.N:
        N = args.N
    timer = pk.Timer()

    # generate random data on the host
    input_ = host_vector(int)(N)
    # for i in range(N):
    #   sum_ = 0
    #   for j in range(S):
    #     sum_ += dist(rng)
    #   input_[i] = sum_ / S
    init_histogram_vector(input_, S)
  
    init_time = timer.seconds()
    timer.reset()

    for i in range(args.M):
        # demonstrate dense histogram method
        print("Dense Histogram")
        histogram = device_vector(int)()
        dense_histogram(input_, histogram)
       
      
        # demonstrate sparse histogram method
        print("Sparse Histogram")
        histogram_values = device_vector(int)()
        histogram_counts = device_vector(int)()
        sparse_histogram(input_, histogram_values, histogram_counts)

    time = timer.seconds()

    # outupt timing info
    if args.file:
        with open(args.file, "a") as f:
            # workload, cuda, size1, size2, time
            f.write("%s, %d, %d, %d, %f, %f\n" % ("histogram", args.cuda, N, 1, init_time, time))


if __name__ == "__main__":
    run()
