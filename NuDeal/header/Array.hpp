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
	if (HostSide::empty()) {
		ndim = rhs.ndim;
		stride = rhs.stride;
		HostSide::_create();
	}
	std::copy(std::execution::par, rhs.begin(), rhs.end(), begin());
}

template <typename T, unsigned N>
void Array_t<T, N, is_portable_t<T>>::CopyD(const Array_t<T, N>& rhs)
{
	if (device_empty()) _cudamalloc(rhs.ndim, rhs.stride);
	cudaCheckError(
		cudaMemcpy(device_data(), rhs.device_data(), device_size() * sizeof(T), cudaMemcpyDeviceToDevice)
	);
}

template <typename T, unsigned N>
void Array_t<T, N, is_portable_t<T>>::CopyHtoD()
{
	if (device_empty()) _cudamalloc();
	cudaCheckError(
		cudaMemcpy(device_data(), data(), device_size() * sizeof(T), cudaMemcpyHostToDevice)
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
	size_type sz = end ? end - begin : device_size();
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
	size_type sz = end ? end - begin : device_size();
	cudaCheckError(
		cudaMemcpy(device_data(), begin, sz * sizeof(T), cudaMemcpyDeviceToDevice)
	);
}

template <typename T, unsigned N>
void Array_t<T, N, is_portable_t<T>>::FillDevice(const_reference val)
{
	if (device_empty()) return;

	T* ptr;
	if (empty()) {
		ptr = new T[device_size()];
		std::fill(std::execution::par, ptr, ptr + device_size(), val);
	}
	else ptr = data();

	cudaCheckError(
		cudaMemcpy(device_data(), ptr, device_size() * sizeof(T), cudaMemcpyHostToDevice)
	);

	if (empty()) delete[] ptr;
}

template <typename T, unsigned N>
template <typename... Ts, typename>
void Array_t<T, N, is_portable_t<T>>::Reshape(size_type first, Ts... pack)
{
	HostSide::Reshape(first, pack...);
	
	if (device_empty()) return;

	auto _n = device_size();

	ndim = 1 + sizeof...(pack);
	stride.front() = first;
	int i = 1; (..., (stride[i++] = stride[i - 1] * pack));

#ifdef _DEBUG
	assert(_n == device_size());
#endif
}

}