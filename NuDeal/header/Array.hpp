#pragma once
#include "Array.h"

namespace LinPack
{

template <typename T, unsigned N>
template <typename... Ts, typename>
void Array_t<T, N, is_portable_t<T>>::Create(size_type first, Ts... pack)
{
	HostSide::Create(first, pack...);
}

template <typename T, unsigned N>
template <typename... Ts, typename>
void Array_t<T, N, is_portable_t<T>>::CreateDevice(size_type first, Ts... pack)
{
	ndim = 1 + sizeof...(Ts);
	stride.front() = first;
	int i = 1; (..., (stride[i++] = stride[i - 1] * pack));
	_cudamalloc();
}

template <typename T, unsigned N>
void Array_t<T, N, is_portable_t<T>>::CopyH(const Array_t<T, N>& rhs)
{
	this->HostSide::operator=(rhs);
}

template <typename T, unsigned N>
void Array_t<T, N, is_portable_t<T>>::CopyD(const Array_t<T, N>& rhs)
{
	ndim = rhs.ndim;
	stride = rhs.stride;
	_cudamalloc();
	cudaCheckError(
		cudaMemcpy(device_data(), rhs.device_data(), size() * sizeof(T), cudaMemcpyDeviceToDevice)
	);
}

template <typename T, unsigned N>
void Array_t<T, N, is_portable_t<T>>::CopyHtoD()
{
	if (device_empty()) _cudamalloc();
	cudaCheckError(
		cudaMemcpy(device_data(), data(), size() * sizeof(T), cudaMemcpyHostToDevice)
	);
}

template <typename T, unsigned N>
void Array_t<T, N, is_portable_t<T>>::CopyDtoH()
{
	if (empty()) HostSide::_create();
	cudaCheckError(
		cudaMemcpy(data(), device_data(), size() * sizeof(T), cudaMemcpyDeviceToHost)
	);
}

template <typename T, unsigned N>
void Array_t<T, N, is_portable_t<T>>::CopyHtoD(const_pointer begin, const_pointer end)
{
	size_type sz = end ? end - begin : size();
	cudaCheckError(
		cudaMemcpy(device_data(), begin, sz * sizeof(T), cudaMemcpyHostToDevice)
	);
}

template <typename T, unsigned N>
void Array_t<T, N, is_portable_t<T>>::CopyDtoH(const_pointer begin, const_pointer end)
{
	size_type sz = end ? end - begin : size();
	cudaCheckError(
		cudaMemcpy(data(), begin, sz * sizeof(T), cudaMemcpyDeviceToHost)
	);
}

template <typename T, unsigned N>
void Array_t<T, N, is_portable_t<T>>::CopyHtoH(const_pointer begin, const_pointer end)
{
	size_type sz = end ? end - begin : size();
	std::copy(std::execution::par, begin, begin + sz, this->begin());
}

template <typename T, unsigned N>
void Array_t<T, N, is_portable_t<T>>::CopyDtoD(const_pointer begin, const_pointer end)
{
	size_type sz = end ? end - begin : size();
	cudaCheckError(
		cudaMemcpy(device_data(), begin, sz * sizeof(T), cudaMemcpyDeviceToDevice)
	);
}

template <typename T, unsigned N>
void Array_t<T, N, is_portable_t<T>>::FillDevice(const_reference val)
{
	if (device_empty()) return;

	std::unique_ptr<array_type> host_ptr(new T[size()]);
	T* ptr = host_ptr.get();
	std::fill(std::execution::par, ptr, ptr + size(), val);

	cudaCheckError(
		cudaMemcpy(device_data(), ptr, size() * sizeof(T), cudaMemcpyHostToDevice)
	);
}

template <typename T, unsigned N>
template <typename... Ts, typename>
void Array_t<T, N, is_portable_t<T>>::Reshape(size_type first, Ts... pack)
{
	HostSide::Reshape(first, pack...);
	
	if (device_empty()) return;

	auto _n = size();

	std::fill(stride.begin(), stride.end(), 0);

	ndim = 1 + sizeof...(pack);
	stride.front() = first;
	int i = 1; (..., (stride[i++] = stride[i - 1] * pack));

#ifdef _DEBUG
	assert(_n == size());
#endif
}

}