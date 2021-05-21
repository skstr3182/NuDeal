#pragma once
#include "Array.h"

namespace LinPack
{

template <typename T>
void Array_t<T>::SetDimension(size_type nx, size_type ny, size_type nz, size_type nw)
{
	this->nx = nx;
	this->ny = ny;
	this->nz = nz;
	this->nw = nw;
	this->nxy = nx * ny;
	this->nxyz = nx * ny * nz;
	this->n = nx * ny * nz * nw;
}

template <typename T> template <typename U>
Array_t<T>::Array_t(const Array_t<U>& rhs)
{
	hostState = static_cast<State>(rhs.hostState);
	ResizeHost(rhs.nx, rhs.ny, rhs.nz, rhs.nw);
	#pragma omp parallel for schedule(guided)
	for (index_type i = 0; i < n; ++i) Entry[i] = static_cast<T>(rhs.Entry[i]);
}

template <typename T>
Array_t<T>::Array_t(const Array_t<T>& rhs)
{
	hostState = rhs.hostState;
	ResizeHost(rhs.nx, rhs.ny, rhs.nz, rhs.nw);
	std::copy(rhs.begin(), rhs.end(), begin());
}

template <typename T>
void Array_t<T>::Alias(const Array_t<T>& rhs)
{
	SetDimension(rhs.nx, rhs.ny, rhs.nz, rhs.nw);
	Entry = rhs.Entry; hostState = State::Alias;
	d_Entry = rhs.d_Entry; devState = State::Alias;
}

template <typename T>
void Array_t<T>::AliasHost(const Array_t<T>& rhs)
{
	SetDimension(rhs.nx, rhs.ny, rhs.nz, rhs.nw);
	Entry = rhs.Entry; hostState = State::Alias;
}

template <typename T>
void Array_t<T>::AliasHost(pointer ptr, size_type nx, size_type ny, size_type nz, size_type nw)
{
	SetDimension(nx, ny, nz, nw);
	Entry = rhs.Entry; hostState = State::Alias;
}

template <typename T>
void Array_t<T>::AliasDevice(const Array_t<T>& rhs)
{
	SetDimension(rhs.nx, rhs.ny, rhs.nz, rhs.nw);
	d_Entry = rhs.d_Entry; devState = State::Alias;
}

template <typename T>
void Array_t<T>::AliasDevice(pointer ptr, size_type nx, size_type ny, size_type nz, size_type nw)
{
	SetDimension(nx, ny, nz, nw);
	d_Entry = ptr; devState = State::Alias;
}

template <typename T>
void Array_t<T>::ResizeHost(size_type nx, size_type ny, size_type nz, size_type nw)
{
	ClearHost();
	SetDimension(nx, ny, nz, nw);
	Entry = new T[n]; hostState = State::Alloc;
}

template <typename T>
void Array_t<T>::ResizeHost(const_pointer ptr, size_type nx, size_type ny, size_type nz, size_type nw)
{
	ClearHost();
	SetDimension(nx, ny, nz, nw);
	Entry = new T[n]; hostState = State::Alloc;
	std::copy(ptr, ptr + n, begin());
}

template <typename T>
void Array_t<T>::ResizeDevice(size_type nx, size_type ny, size_type nz, size_type nw)
{
	ClearDevice();
	SetDimension(nx, ny, nz, nw);
	cudaMalloc(&d_Entry, n * sizeof(T)); devState = State::Alloc;
}

template <typename T>
void Array_t<T>::ResizeDevice(const_pointer ptr, size_type nx, size_type ny, size_type nz, size_type nw)
{
	ClearDevice();
	SetDimension(nx, ny, nz, nw);
	cudaMalloc(&d_Entry, n * sizeof(T)); devState = State::Alloc;
	cudaMemcpy(d_Entry, ptr, n * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void Array_t<T>::ClearHost()
{
	if (hostState == State::Alloc) delete[] Entry;
	Entry = static_cast<pointer>(NULL);
	hostState = State::Undefined;
}

template <typename T>
void Array_t<T>::ClearDevice()
{
	if (devState == State::Alloc) cudaFree(d_Entry);
	d_Entry = static_cast<pointer>(NULL);
	devState = State::Undefined;
}

}