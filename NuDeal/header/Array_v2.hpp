#pragma once
#include "Array_v2.h"

namespace LinPack_v2
{

namespace // Anonymous Namespace
{

template <typename T>
void ArrayBase_t<T>::Alias(const ArrayBase_t<T>& rhs)
{
	SetDimension(rhs.nx, rhs.ny, rhs.nz, rhs.nw);
	container_host.clear();
	ptr_host = rhs.ptr_host;
}

template <typename T> inline
ArrayBase_t<T>& ArrayBase_t<T>::operator+=(const_reference val)
{
	#pragma omp parallel for schedule(guided)
	for (index_type i = 0; i < size(); ++i) container_host[i] += val;
	return *this;
}

template <typename T> template <typename U> inline
ArrayBase_t<T>& ArrayBase_t<T>::operator+=(const ArrayBase_t<U>& rhs)
{
	#pragma omp parallel for schedule(guided)
	for (index_type i = 0; i < size(); ++i) container_host[i] += static_cast<T>(rhs[i]);
	return *this;
}

template <typename T> inline
ArrayBase_t<T>& ArrayBase_t<T>::operator+=(const ArrayBase_t<T>& rhs)
{
	#pragma omp parallel for schedule(guided)
	for (index_type i = 0; i < size(); ++i) container_host[i] += rhs[i];
	return *this;
}

template <typename T> inline
ArrayBase_t<T>& ArrayBase_t<T>::operator-=(const_reference val)
{
	#pragma omp parallel for schedule(guided)
	for (index_type i = 0; i < size(); ++i) container_host[i] -= val;
	return *this;
}

template <typename T> template <typename U> inline
ArrayBase_t<T>& ArrayBase_t<T>::operator-=(const ArrayBase_t<U>& rhs)
{
	#pragma omp parallel for schedule(guided)
	for (index_type i = 0; i < size(); ++i) container_host[i] -= static_cast<T>(rhs[i]);
	return *this;
}

template <typename T> inline
ArrayBase_t<T>& ArrayBase_t<T>::operator-=(const ArrayBase_t<T>& rhs)
{
	#pragma omp parallel for schedule(guided)
	for (index_type i = 0; i < size(); ++i) container_host[i] -= rhs[i];
	return *this;
}

template <typename T> inline
ArrayBase_t<T>& ArrayBase_t<T>::operator*=(const_reference val)
{
	#pragma omp parallel for schedule(guided)
	for (index_type i = 0; i < size(); ++i) container_host[i] *= val;
	return *this;
}

template <typename T> template <typename U> inline
ArrayBase_t<T>& ArrayBase_t<T>::operator*=(const ArrayBase_t<U>& rhs)
{
	#pragma omp parallel for schedule(guided)
	for (index_type i = 0; i < size(); ++i) container_host[i] *= static_cast<T>(rhs[i]);
	return *this;
}

template <typename T> inline
ArrayBase_t<T>& ArrayBase_t<T>::operator*=(const ArrayBase_t<T>& rhs)
{
	#pragma omp parallel for schedule(guided)
	for (index_type i = 0; i < size(); ++i) container_host[i] *= rhs[i];
	return *this;
}

template <typename T> inline
ArrayBase_t<T>& ArrayBase_t<T>::operator/=(const_reference val)
{
	#pragma omp parallel for schedule(guided)
	for (index_type i = 0; i < size(); ++i) container_host[i] /= val;
	return *this;
}

template <typename T> template <typename U> inline
ArrayBase_t<T>& ArrayBase_t<T>::operator/=(const ArrayBase_t<U>& rhs)
{
	#pragma omp parallel for schedule(guided)
	for (index_type i = 0; i < size(); ++i) container_host[i] /= static_cast<T>(rhs[i]);
	return *this;
}

template <typename T> inline
ArrayBase_t<T>& ArrayBase_t<T>::operator/=(const ArrayBase_t<T>& rhs)
{
	#pragma omp parallel for schedule(guided)
	for (index_type i = 0; i < size(); ++i) container_host[i] /= rhs[i];
	return *this;
}

} // Anonymous Namespace


}