/* Matrix transpose for arbitrary datatypes (e.g. Cartesian<double>)
 * Note that cuBLAS geam outperforms this function for most datatypes
 * Cloned from http://www.orangeowlsolutions.com/archives/790
 */

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
namespace transpose {
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////


struct index : public thrust::unary_function<size_t,size_t>
{
  const size_t m, n;

  __host__ __device__
  index(size_t m, size_t n) : m(m), n(n) {}

  __host__ __device__
  size_t operator()(size_t ix)
  {
    size_t i = ix / n;
    size_t j = ix % n;

    return m * j + i;
  }
};

template <typename T = double, typename Iterator>
void transpose(size_t m, size_t n, const thrust::device_vector<T>& src, Iterator dst)
{
  thrust::counting_iterator<size_t> indices(0);
  thrust::gather
    (thrust::make_transform_iterator(indices, transpose::index(n, m)),
     thrust::make_transform_iterator(indices, transpose::index(n, m)) + src.size(),
     src.begin(),
     dst);
}


  ///////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////
} // end namespace
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
