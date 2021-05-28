#include <thrust/device_vector.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <thrust/iterator/discard_iterator.h>
#include <iostream>

#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <fstream>
#include <sys/time.h>

// This example illustrates use of the set operation algorithms
//  - merge
//  - set_union
//  - set_intersection
//  - set_difference
//  - set_symmetric_difference
//
// In this context a "set" is simply a sequence of sorted values,
// allowing the standard set operations to be performed more efficiently
// than on unsorted data.  Since the output of a set operation is a valid
// set (i.e. a sorted sequence) it is possible to apply the set operations
// in a nested fashion to compute arbitrary set expressions.
//
// Set operation usage notes:
//   - The output set size is variable (except for thrust::merge),
//     so the return value is important.
//   - Generally one would conservatively allocate storage for the output
//     and then resize or shrink an output container as necessary.
//     Alternatively, one can compute the exact output size by
//     outputting to a discard_iterator.  This approach is more computationally
//     expensive (approximately 2x), but conserves memory capacity.
//     Refer to the SetIntersectionSize function for implementation details.
//   - Sets are allowed to have duplicate elements, which are carried
//     through to the output in a algorithm-specific manner.  Refer
//     to the full documentation for precise semantics.


// helper routine
template <typename String, typename Vector>
void print(const String& s, const Vector& v)
{
  std::cout << s << " [";
  // for(size_t i = 0; i < v.size(); i++)
  //   std::cout << " " << v[i];
  std::cout << " ]\n";
}

template <typename Vector>
void Merge(const Vector& A, const Vector& B)
{
  // merged output is always exactly A.size() + B.size()
  Vector C(A.size() + B.size());

  thrust::merge(A.begin(), A.end(), B.begin(), B.end(), C.begin());

  print("Merge(A,B)", C);
}

template <typename Vector>
void SetUnion(const Vector& A, const Vector& B)
{
  // union output is at most A.size() + B.size()
  Vector C(A.size() + B.size());

  // set_union returns an iterator C_end denoting the end of input
  typename Vector::iterator C_end;
  
  C_end = thrust::set_union(A.begin(), A.end(), B.begin(), B.end(), C.begin());
  
  // shrink C to exactly fit output
  C.erase(C_end, C.end());

  print("Union(A,B)", C);
}

template <typename Vector>
void SetIntersection(const Vector& A, const Vector& B)
{
  // intersection output is at most min(A.size(), B.size())
  Vector C(thrust::min(A.size(), B.size()));

  // set_union returns an iterator C_end denoting the end of input
  typename Vector::iterator C_end;
  
  C_end = thrust::set_intersection(A.begin(), A.end(), B.begin(), B.end(), C.begin());
  
  // shrink C to exactly fit output
  C.erase(C_end, C.end());

  print("Intersection(A,B)", C);
}

template <typename Vector>
void SetDifference(const Vector& A, const Vector& B)
{
  // difference output is at most A.size()
  Vector C(A.size());

  // set_union returns an iterator C_end denoting the end of input
  typename Vector::iterator C_end;
  
  C_end = thrust::set_difference(A.begin(), A.end(), B.begin(), B.end(), C.begin());
  
  // shrink C to exactly fit output
  C.erase(C_end, C.end());

  print("Difference(A,B)", C);
}

template <typename Vector>
void SetSymmetricDifference(const Vector& A, const Vector& B)
{
  // symmetric difference output is at most A.size() + B.size()
  Vector C(A.size() + B.size());

  // set_union returns an iterator C_end denoting the end of input
  typename Vector::iterator C_end;
  
  C_end = thrust::set_symmetric_difference(A.begin(), A.end(), B.begin(), B.end(), C.begin());
  
  // shrink C to exactly fit output
  C.erase(C_end, C.end());

  print("SymmetricDifference(A,B)", C);
}

template <typename Vector>
void SetIntersectionSize(const Vector& A, const Vector& B)
{
  // computes the exact size of the intersection without allocating output
  thrust::discard_iterator<> C_begin, C_end;

  C_end = thrust::set_intersection(A.begin(), A.end(), B.begin(), B.end(), C_begin);

  std::cout << "SetIntersectionSize(A,B) " << (C_end - C_begin) << std::endl;
}


int main(int argc, char *argv[])
{

  int N = atoi(argv[1]);

  struct timeval begin, end;
  gettimeofday( &begin, NULL );

  // int a[] = {0,2,4,5,6,8,9};
  // int b[] = {0,1,2,3,5,7,8};

  // thrust::device_vector<int> A(a, a + sizeof(a) / sizeof(int));
  // thrust::device_vector<int> B(b, b + sizeof(b) / sizeof(int));

  // generate random data on the host
  thrust::device_vector<int> A(N);
  thrust::device_vector<int> B(N);

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

  print("Set A", A);
  print("Set B", B);

  gettimeofday( &end, NULL );
  // Calculate time.
  double init_time = 1.0 * ( end.tv_sec  - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );
  gettimeofday( &begin, NULL );

  for (int i = 0; i < atoi(argv[2]); i++) {
    Merge(A,B);
    SetUnion(A,B);
    SetIntersection(A,B);
    SetDifference(A,B);
    SetSymmetricDifference(A,B);

    SetIntersectionSize(A,B);
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
  outfile << "set" << "," << CUDA << "," << N << "," << 1 << "," 
    << init_time << "," << time << std::endl;;

  return 0;
}
