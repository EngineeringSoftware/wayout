#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

#include <fstream>
#include <sys/time.h>

int main(int argc, char *argv[])
{
  int N = atoi(argv[1]);

  struct timeval begin, end;
  gettimeofday( &begin, NULL );

  // generate 32M random numbers serially
  // thrust::host_vector<int> h_vec(32 << 20);
  thrust::host_vector<int> h_vec(N);
  std::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer data to the device
  thrust::device_vector<int> d_vec = h_vec;

  gettimeofday( &end, NULL );
  // Calculate time.
  double init_time = 1.0 * ( end.tv_sec  - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );
  gettimeofday( &begin, NULL );

  for (int i = 0; i < atoi(argv[2]); i++) {

    // sort data on the device (846M keys per second on GeForce GTX 480)
    thrust::sort(d_vec.begin(), d_vec.end());
  }

  // transfer data back to host
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

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
  outfile << "sort" << "," << CUDA << "," << N << "," << 1 << "," 
    << init_time << "," << time << std::endl;;

  return 0;
}
