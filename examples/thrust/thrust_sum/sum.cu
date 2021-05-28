#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>

#include <fstream>
#include <sys/time.h>

int my_rand(void)
{
  static thrust::default_random_engine rng;
  static thrust::uniform_int_distribution<int> dist(0, 9999);
  return dist(rng);
}

int main(int argc, char *argv[])
{
  int N = atoi(argv[1]);

  struct timeval begin, end;
  gettimeofday( &begin, NULL );

  // generate random data on the host
  // thrust::host_vector<int> h_vec(100);
  thrust::host_vector<int> h_vec(N);
  thrust::generate(h_vec.begin(), h_vec.end(), my_rand);

  // transfer to device and compute sum
  thrust::device_vector<int> d_vec = h_vec;

  gettimeofday( &end, NULL );
  // Calculate time.
  double init_time = 1.0 * ( end.tv_sec  - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );
  gettimeofday( &begin, NULL );

  int sum;
  for (int i = 0; i < atoi(argv[2]); i++) {

    // initial value of the reduction
    int init = 0; 
   
    // binary operation used to reduce values
    thrust::plus<int> binary_op;

    // compute sum on the device
    sum += thrust::reduce(d_vec.begin(), d_vec.end(), init, binary_op);
  }

  // print the sum
  std::cout << "sum is " << sum << std::endl;

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
  outfile << "sum" << "," << CUDA << "," << N << "," << 1 << "," 
    << init_time << "," << time << std::endl;;
 

  return 0;
}
