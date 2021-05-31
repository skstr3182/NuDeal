#pragma once
#include "Array.h"

namespace LinPack
{

template <typename T>
void Array_t<T>::Resize(size_type nx, size_type ny, size_type nz, size_type nw)
{
	auto _unique_host = std::move(unique_host);
	auto _L = std::move(L);

	L(nx, ny, nz, nw);
	unique_host = std::make_unique<array_type>(L.n);
	if (entry_host) {
		auto beg = entry_host;
		auto end = entry_host + min(L.n, _L.n);
		std::copy(std::execution::par, beg, end, unique_host.get());
		_unique_host.reset();
	}

	entry_host = unique_host.get();
}

template <typename T>
void Array_t<T>::ResizeDevice()
{
	unique_device.reset();
	unique_device = std::unique_ptr<array_type, cudaDeleter_t>
		(Array_t<T>::cuda_allocator(L.n), Array_t<T>::cuda_deleter);
	entry_device = unique_device.get();
}

template <typename T>
void Array_t<T>::ResizeDevice(size_type nx, size_type ny, size_type nz, size_type nw)
{
	if (HasHost()) { 
		if (!L.check(nx, ny, nz, nw)) 
			exit(EXIT_FAILURE); // Exception
	}
	L(nx, ny, nz, nw);
	ResizeDevice();
}

template <typename T>
void Array_t<T>::CopyHtoD()
{
	cudaCheckError(
		cudaMemcpy(dev_ptr(), host_ptr(), size() * sizeof(T), cudaMemcpyHostToDevice)
	);
}

template <typename T>
void Array_t<T>::CopyDtoH()
{
	cudaCheckError(
		cudaMemcpy(host_ptr(), dev_ptr(), size() * sizeof(T), cudaMemcpyDeviceToHost)
	);
}

template <typename T>
void Array_t<T>::Copy(const Array_t<T>& rhs)
{
	Clear();
	L = rhs.L;

	if (rhs.HasHost()) {
		unique_host = std::make_unique<array_type>(L.n);
		entry_host = unique_host.get();
		std::copy(std::execution::par, rhs.begin(), rhs.end(), host_ptr());
	}

	if (rhs.HasDevice()) {
		unique_device = std::unique_ptr<array_type, cudaDeleter_t>
			(Array_t<T>::cuda_allocator(L.n), Array_t<T>::cuda_deleter);
		entry_device = unique_device.get();
		cudaCheckError(
			cudaMemcpy(dev_ptr(), rhs.dev_ptr(), size() * sizeof(T), cudaMemcpyDeviceToDevice)
		);
	}
}

template <typename T>
void Array_t<T>::Copy(const_pointer begin, const_pointer end)
{
	cudaPointerAttributes attr;
	cudaCheckError(cudaPointerGetAttributes(&attr, begin));

	size_type sz;
	if (end) sz = end - begin;
	else sz = size();

	if (attr.type == cudaMemoryTypeUnregistered)
		std::copy(std::execution::par, begin, begin + sz, host_ptr());
	else if (attr.type == cudaMemoryTypeDevice)
		cudaCheckError(
			cudaMemcpy(dev_ptr(), begin, sz * sizeof(T), cudaMemcpyDeviceToDevice)
		);
}

template <typename T>
void Array_t<T>::Alias(Array_t<T>& rhs)
{
	Clear();
	L = rhs.L;
	entry_host = rhs.host_ptr();
	entry_device = rhs.dev_ptr();
}

template <typename T>
void Array_t<T>::Alias(pointer ptr, size_type nx, size_type ny, size_type nz, size_type nw)
{
	cudaPointerAttributes attr;
	cudaCheckError(cudaPointerGetAttributes(&attr, ptr));

	Clear();
	L(nx, ny, nz, nw);

	if (attr.type == cudaMemoryTypeUnregistered) 
		entry_host = ptr;
	else if (attr.type == cudaMemoryTypeDevice)
		entry_device = ptr;
}

template <typename T>
void Array_t<T>::Fill(const_reference val)
{
	std::fill(std::execution::par, begin(), end(), val);
}

}