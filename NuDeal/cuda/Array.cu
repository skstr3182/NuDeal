#include "Array.h"
#include "Array.hpp"

namespace LinPack
{

template <typename T>
__global__ void _Fill(size_t n, T val, T *ptr)
{
	size_t thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread_idx >= n) return;
	ptr[thread_idx] = val;
}

template <typename T>
void Array_t<T>::Fill(const_reference val)
{
	dim3 threads(1024, 1, 1);
	dim3 blocks(size() / threads.x + 1, 1, 1);

	_Fill <<< blocks, threads >>> (size(), val, dev_ptr());

	std::fill(std::execution::par, begin(), end(), val);
}

template class Array_t<bool>;
template class Array_t<char>;
template class Array_t<int>;
template class Array_t<int2>;
template class Array_t<int3>;
template class Array_t<int4>;
template class Array_t<float>;
template class Array_t<float2>;
template class Array_t<float3>;
template class Array_t<float4>;
template class Array_t<double>;
template class Array_t<double2>;
template class Array_t<double3>;
template class Array_t<double4>;

}