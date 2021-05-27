#pragma once
#include "Defines.h"
#include <thrust/device_vector.h>

namespace LinPack
{

template <typename T>
class Vector_t
{
template <typename U> friend class Vector_t;
public :

	using reference       = T&;
	using const_reference = const T&;
	using size_type       = size_t;
	using index_type			= long long;
	using value_type      = T;
	using pointer         = T*;
	using const_pointer   = const T*;
	using iterator        = typename vector<T>::iterator;
	using const_iterator  = typename vector<T>::const_iterator;
	using dimension_t			= unsigned int;

private:

	dimension_t ndim = 0, dims[10] = { 0, };
	std::vector<T> buf_host;
	thrust::device_vector<T> buf_device;
	pointer ptr_host = static_cast<pointer>(NULL);
	pointer ptr_device = static_cast<pointer>(NULL);

public: // Constructor & Destructor

	Vector_t() = default;
	template <typename U> Vector_t(const Vector_t<U>& rhs);
	Vector_t(const Vector_t<T>& rhs);
	Vector_t(Vector_t<T>&& rhs);
	explicit Vector_t(size_type size...);
	explicit Vector_t(const_pointer ptr, size_type size...);

public: // Resizing

	void ResizeHost(std::initializer_list<size_type> list);
	void ResizeHost(size_type n...);
	void ResizeHost(const_pointer ptr, size_type n...);
	void ResizeDevice(size_type n...);
	void ResizeDevice(const_pointer ptr, size_type n...);


};

}