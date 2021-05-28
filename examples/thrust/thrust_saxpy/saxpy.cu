#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <iostream>
#include <iterator>
#include <algorithm>

#include <thrust/random.h>
#include <fstream>
#include <sys/time.h>

// This example illustrates how to implement the SAXPY
// operation (Y[i] = a * X[i] + Y[i]) using Thrust. 
// The saxpy_slow function demonstrates the most
// straightforward implementation using a temporary
// array and two separate transformations, one with
// multiplies and one with plus.  The saxpy_fast function
// implements the operation with a single transformation
// and represents "best practice".

struct saxpy_functor : public thrust::binary_function<double,double,double>
{
    const double a;

    saxpy_functor(double _a) : a(_a) {}

    __host__ __device__
        double operator()(const double& x, const double& y) const { 
            return a * x + y;
        }
};

void saxpy_fast(double A, thrust::device_vector<double>& X, thrust::device_vector<double>& Y)
{
    // Y <- A * X + Y
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void saxpy_slow(double A, thrust::device_vector<double>& X, thrust::device_vector<double>& Y)
{
    thrust::device_vector<double> temp(X.size());
   
    // temp <- A
    thrust::fill(temp.begin(), temp.end(), A);
    
    // temp <- A * X
    thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<double>());

    // Y <- A * X + Y
    thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<double>());
}

int main(int argc, char *argv[])
{
    int N = atoi(argv[1]);

    struct timeval begin, end;
    gettimeofday( &begin, NULL );


    // initialize host arrays
    // double x[4] = {1.0, 1.0, 1.0, 1.0};
    // double y[4] = {1.0, 2.0, 3.0, 4.0};

    thrust::host_vector<double> x(N), y(N);
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<double> dist(0, x.size());

    for(size_t i = 0; i < x.size(); i++) {
      x[i] = dist(rng);
      y[i] = dist(rng);
    }
    thrust::device_vector<double> X(x);
    thrust::device_vector<double> Y(y);

    gettimeofday( &end, NULL );
    // Calculate time.
    double init_time = 1.0 * ( end.tv_sec  - begin.tv_sec ) +
                  1.0e-6 * ( end.tv_usec - begin.tv_usec );
    gettimeofday( &begin, NULL );

    for (int i = 0; i < atoi(argv[2]); i++) {
      {
          // transfer to device
          // thrust::device_vector<double> X(x, x + 4);
          // thrust::device_vector<double> Y(y, y + 4);
        
          // slow method
          saxpy_slow(2.0, X, Y);
      }

      // {
      //     // transfer to device
      //     thrust::device_vector<double> X(x, x + 4);
      //     thrust::device_vector<double> Y(y, y + 4);

      //     // fast method
      //     saxpy_fast(2.0, X, Y);
      // }
    }

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
    outfile << "saxpy" << "," << CUDA << "," << N << "," << 1 << "," 
      << init_time << "," << time << std::endl;;
 
    
    return 0;
}

