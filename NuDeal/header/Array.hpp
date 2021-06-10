#pragma once
#include "Array.h"

namespace LinPack
{

template <typename T>
void ArrayBase_t<T>::Resize(size_type nx, size_type ny, size_type nz, size_type nw)
{
	auto _L = std::move(L);

	L(nx, ny, nz, nw);

	if (L.n != _L.n) {
		auto _unique_host = std::move(unique_host);
		unique_host = std::make_unique<array_type>(L.n);
		std::copy(std::execution::par, entry_host, entry_host + min(L.n, _L.n), unique_host.get());
	}

	entry_host = unique_host.get();
}

template <typename T>
void ArrayBase_t<T>::Copy(const ArrayBase_t<T>& rhs)
{
	Clear();
	L = rhs.L;

	unique_host = std::make_unique<array_type>(L.n);
	entry_host = unique_host.get();
	std::copy(std::execution::par, rhs.begin(), rhs.end(), host_ptr());
}

template <typename T>
void ArrayBase_t<T>::Copy(const_pointer begin, const_pointer end)
{
	size_type sz = end ? end - begin : size();
	std::copy(std::execution::par, begin, begin + sz, host_ptr());
}

template <typename T>
void ArrayBase_t<T>::Alias(ArrayBase_t<T>& rhs)
{
	Clear();
	_copy(rhs);
}

template <typename T>
void ArrayBase_t<T>::Alias(pointer ptr, size_type nx, size_type ny, size_type nz, size_type nw)
{
	Clear();
	L(nx, ny, nz, nw);
	entry_host = ptr;
}

template <typename T>
void ArrayBase_t<T>::Fill(const_reference val)
{
	std::fill(std::execution::par, begin(), end(), val);
}

template <typename T>
void Array_t<T, is_device_t<T>>::ResizeHost(size_type nx, size_type ny, size_type nz, size_type nw)
{
	MyBase::Resize(nx, ny, nz, nw);
}

template <typename T>
void Array_t<T, is_device_t<T>>::ResizeDevice(size_type nx, size_type ny, size_type nz, size_type nw)
{
	auto _L = std::move(L);

	L(nx, ny, nz, nw);

	if (L.n != _L.n) {
		auto _unique_device = std::move(unique_device);
		unique_device = make_device_ptr(device_size());
		cudaCheckError(
			cudaMemcpy(unique_device.get(), entry_device, min(L.n, _L.n) * sizeof(T), cudaMemcpyDeviceToDevice)
		);
	}

	entry_device = unique_device.get();
}

template <typename T>
void Array_t<T, is_device_t<T>>::ResizeHost()
{
	if (L.n == MyBase::L.n)
		MyBase::L = L;
	else 
		MyBase::Resize(L.nx, L.ny, L.nz, L.nw);
}

template <typename T>
void Array_t<T, is_device_t<T>>::ResizeDevice()
{
	if (L.n == MyBase::L.n)
		L = MyBase::L;
	else
		ResizeDevice(MyBase::L.nx, MyBase::L.ny, MyBase::L.nz, MyBase::L.nw);
}

template <typename T>
void Array_t<T, is_device_t<T>>::Resize(size_type nx, size_type ny, size_type nz, size_type nw)
{
	ResizeHost(nx, ny, nz, nw);
	ResizeDevice(nx, ny, nz, nw);
}

template <typename T>
void Array_t<T, is_device_t<T>>::Copy(const Array_t<T>& rhs)
{
	Clear();

	MyBase::Copy(rhs);

	L = rhs.L;
	unique_device = make_device_ptr(device_size());
	entry_device = unique_device.get();
	cudaCheckError(
		cudaMemcpy(entry_device, rhs.device_ptr(), L.n * sizeof(T), cudaMemcpyDeviceToDevice)
	);
}

template <typename T>
void Array_t<T, is_device_t<T>>::Copy(const_pointer begin, const_pointer end)
{
	cudaPointerAttributes attr;
	cudaCheckError(cudaPointerGetAttributes(&attr, begin));

	size_type sz;
	if (attr.type == cudaMemoryTypeDevice) {
		sz = end ? end - begin : device_size();
		cudaCheckError(
			cudaMemcpy(entry_device, begin, sz * sizeof(T), cudaMemcpyDeviceToDevice)
		);
	}
	else {
		sz = end ? end - begin : host_size();
		std::copy(std::execution::par, begin, begin + sz, entry_host);
	}
}

template <typename T>
void Array_t<T, is_device_t<T>>::CopyHtoD()
{
	if (device_size() < host_size()) ResizeDevice();
	cudaCheckError(
		cudaMemcpy(entry_device, entry_host, host_size() * sizeof(T), cudaMemcpyHostToDevice)
	);
}

template <typename T>
void Array_t<T, is_device_t<T>>::CopyHtoD(const_pointer begin, const_pointer end)
{
	size_t sz = end ? end - begin : device_size();
	cudaCheckError(
		cudaMemcpy(entry_device, begin, sz * sizeof(T), cudaMemcpyHostToDevice)
	);
}

template <typename T>
void Array_t<T, is_device_t<T>>::CopyDtoH()
{
	if (host_size() < device_size()) ResizeHost();
	cudaCheckError(
		cudaMemcpy(entry_host, entry_device, device_size() * sizeof(T), cudaMemcpyDeviceToHost)
	);
}

template <typename T>
void Array_t<T, is_device_t<T>>::CopyDtoH(const_pointer begin, const_pointer end)
{
	size_t sz = end ? end - begin : host_size();
	cudaCheckError(
		cudaMemcpy(entry_host, begin, sz * sizeof(T), cudaMemcpyDeviceToHost)
	);
}

template <typename T>
void Array_t<T, is_device_t<T>>::Alias(Array_t<T>& rhs)
{
	Clear();
	_copy(rhs);
}

template <typename T>
void Array_t<T, is_device_t<T>>::Alias(pointer ptr, size_type nx, size_type ny, size_type nz, size_type nw)
{
	Clear();
	L(nx, ny, nz, nw);
	
	cudaPointerAttributes attr;
	cudaCheckError(cudaPointerGetAttributes(&attr, ptr));

	if (attr.type == cudaMemoryTypeDevice)
		entry_device = ptr;
	else
		entry_host = ptr;
}

}