#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <cassert>
#include <iostream>

#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <fstream>
#include <sys/time.h>

template <typename IndexVector,
          typename ValueVector>
void print_sparse_vector(const IndexVector& A_index,
                         const ValueVector& A_value)
{
    assert(A_index.size() == A_value.size());

    // for(size_t i = 0; i < A_index.size(); i++)
    //     std::cout << "(" << A_index[i] << "," << A_value[i] << ") ";
    std::cout << std::endl;
}

template <typename IndexVector1,
          typename ValueVector1,
          typename IndexVector2,
          typename ValueVector2,
          typename IndexVector3,
          typename ValueVector3>
void sum_sparse_vectors(const IndexVector1& A_index,
                        const ValueVector1& A_value,
                        const IndexVector2& B_index,
                        const ValueVector2& B_value,
                              IndexVector3& C_index,
                              ValueVector3& C_value)
{
    typedef typename IndexVector3::value_type  IndexType;
    typedef typename ValueVector3::value_type  ValueType;

    assert(A_index.size() == A_value.size());
    assert(B_index.size() == B_value.size());

    size_t A_size = A_index.size();
    size_t B_size = B_index.size();

    // allocate storage for the combined contents of sparse vectors A and B
    IndexVector3 temp_index(A_size + B_size);
    ValueVector3 temp_value(A_size + B_size);

    // merge A and B by index
    thrust::merge_by_key(A_index.begin(), A_index.end(),
                         B_index.begin(), B_index.end(),
                         A_value.begin(),
                         B_value.begin(),
                         temp_index.begin(),
                         temp_value.begin());

    // compute number of unique indices
    size_t C_size = thrust::inner_product(temp_index.begin(), temp_index.end() - 1,
                                          temp_index.begin() + 1,
                                          size_t(0),
                                          thrust::plus<size_t>(),
                                          thrust::not_equal_to<IndexType>()) + 1;

    // allocate space for output
    C_index.resize(C_size);
    C_value.resize(C_size);

    // sum values with the same index
    thrust::reduce_by_key(temp_index.begin(), temp_index.end(),
                          temp_value.begin(),
                          C_index.begin(),
                          C_value.begin(),
                          thrust::equal_to<IndexType>(),
                          thrust::plus<ValueType>());
}

int main(int argc, char *argv[])
{
    int N = atoi(argv[1]);

    struct timeval begin, end;
    gettimeofday( &begin, NULL );

    // // initialize sparse vector A with 4 elements
    // thrust::device_vector<int>   A_index(4);
    // thrust::device_vector<double> A_value(4);
    // A_index[0] = 2;  A_value[0] = 10;
    // A_index[1] = 3;  A_value[1] = 60;
    // A_index[2] = 5;  A_value[2] = 20;
    // A_index[3] = 8;  A_value[3] = 40;

    // // initialize sparse vector B with 6 elements
    // thrust::device_vector<int>   B_index(6);
    // thrust::device_vector<double> B_value(6);
    // B_index[0] = 1;  B_value[0] = 50;
    // B_index[1] = 2;  B_value[1] = 30;
    // B_index[2] = 4;  B_value[2] = 80;
    // B_index[3] = 5;  B_value[3] = 30;
    // B_index[4] = 7;  B_value[4] = 90;
    // B_index[5] = 8;  B_value[5] = 10;

    thrust::device_vector<int>   A_index(N), B_index(N);
    thrust::device_vector<double> A_value(N), B_value(N);

    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<double> dist(0, N);

    thrust::host_vector<int>   a_index(N), b_index(N);
    thrust::host_vector<double> a_value(N), b_value(N);
    for (int i = 0; i < N; i++) {
      thrust::uniform_int_distribution<int> idx_dist(i, i+16);
      a_index[i] = idx_dist(rng);
      a_value[i] = dist(rng);
      b_index[i] = idx_dist(rng);
      b_value[i] = dist(rng);
    }
    thrust::copy(a_index.begin(), a_index.end(), A_index.begin());
    thrust::copy(a_value.begin(), a_value.end(), A_value.begin());
    thrust::copy(b_index.begin(), b_index.end(), B_index.begin());
    thrust::copy(b_value.begin(), b_value.end(), B_value.begin());

    // compute sparse vector C = A + B
    thrust::device_vector<int>   C_index;
    thrust::device_vector<double> C_value;

    gettimeofday( &end, NULL );
    // Calculate time.
    double init_time = 1.0 * ( end.tv_sec  - begin.tv_sec ) +
                  1.0e-6 * ( end.tv_usec - begin.tv_usec );
    gettimeofday( &begin, NULL );

    for (int i = 0; i < atoi(argv[2]); i++) {
      sum_sparse_vectors(A_index, A_value, B_index, B_value, C_index, C_value);
    }

    std::cout << "Computing C = A + B for sparse vectors A and B" << std::endl;
    std::cout << "A "; print_sparse_vector(A_index, A_value);
    std::cout << "B "; print_sparse_vector(B_index, B_value);
    std::cout << "C "; print_sparse_vector(C_index, C_value);

    gettimeofday( &end, NULL );
    // Calculate time.
    double time = 1.0 *    ( end.tv_sec  - begin.tv_sec ) +
                  1.0e-6 * ( end.tv_usec - begin.tv_usec );

    std::ofstream outfile;
    outfile.open(argv[3], std::ios_base::app);
    // workload, cuda, size1, size2, init_time, time
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#define CUDA 1
#else
#define CUDA 0
#endif
    outfile << "sparse" << "," << CUDA << "," << N << "," << 1 << "," 
      << init_time << "," << time << std::endl;;
 
}

