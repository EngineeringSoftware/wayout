#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/generate.h>
#include <algorithm>

void init_histogram_vector(thrust::host_vector<int> &input, int S) {
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, 9);

  // generate random data on the host
  for(int i = 0; i < input.size(); i++)
  {
    int sum = 0;
    for (int j = 0; j < S; j++)
      sum += dist(rng);
    input[i] = sum / S;
  }
}

void init_set_vector(thrust::device_vector<int> &A, thrust::device_vector<int> &B) {
  thrust::host_vector<int> a(A.size());
  thrust::host_vector<int> b(B.size());
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, 100);
  for(int i = 0; i < a.size(); i++)
  {
    a[i] = dist(rng);
    b[i] = dist(rng);
  }
  thrust::copy(a.begin(), a.end(), A.begin());
  thrust::copy(b.begin(), b.end(), B.begin());
  thrust::sort(A.begin(), A.end());
  thrust::sort(B.begin(), B.end());
}

void init_mode_vector(thrust::host_vector<int> &h_data, int M) {
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, M - 1);

  for(size_t i = 0; i < h_data.size(); i++)
    h_data[i] = dist(rng);
}

void init_sparse_vector(thrust::device_vector<int> &A_index, thrust::device_vector<double> &A_value,
  thrust::device_vector<int> &B_index, thrust::device_vector<double> &B_value) {
  int N = A_index.size();
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
}

void init_saxpy_vector(thrust::host_vector<double> &x, thrust::host_vector<double> &y) {
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<double> dist(0, x.size());

  for(size_t i = 0; i < x.size(); i++) {
    x[i] = dist(rng);
    y[i] = dist(rng);
  }
}

int my_rand(void)
{
  static thrust::default_random_engine rng;
  static thrust::uniform_int_distribution<int> dist(0, 9999);
  return dist(rng);
}
void init_sum_vector(thrust::host_vector<int> &h_vec) {
  thrust::generate(h_vec.begin(), h_vec.end(), my_rand);
}

void init_sort_vector(thrust::host_vector<int> &h_vec) {
  std::generate(h_vec.begin(), h_vec.end(), rand);
}
