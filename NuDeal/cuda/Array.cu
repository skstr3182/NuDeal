#include "Array.h"
#include "CUDAExcept.h"

namespace LinPack
{

template <typename T>
__global__ void _Fill(size_t n, T value, T *ptr)
{
	auto thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread_idx >= n) return;

	ptr[thread_idx] = value;
}


template <typename T>
void Array_t<T>::FillDevice(const T& value)
{
	dim3 threads(1024, 1, 1);
	dim3 blocks(n / threads.x + 1, 1, 1);

	_Fill <<< blocks, threads >>> (size(), value, GetDevicePointer());

#ifdef CUDA_DEBUG
	cudaCheckError(cudaDeviceSynchronize());
#endif
}

template class Array_t<int>;
template class Array_t<bool>;
template class Array_t<float>;
template class Array_t<double>;

}