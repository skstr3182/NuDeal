#pragma once
#include "Defines.h"
#include <thrust/device_vector.h>

namespace LinPack_v2
{

namespace // Anonymous Namespace
{

template <typename T>
class ArrayBase_t
{
template <typename U> friend class ArrayBase_t;
public : // Type Aliasing

	using reference       = T&;
	using const_reference = const T&;
	using size_type       = size_t;
	using index_type			= long long;
	using value_type      = T;
	using pointer         = T*;
	using const_pointer   = const T*;
	using iterator        = typename vector<T>::iterator;
	using const_iterator  = typename vector<T>::const_iterator;
	using dimension_t			= ulonglong4;

protected:

	size_type n = 0;
	dimension_t dims = {0, 0, 0, 0};
	size_type& nx = dims.x;
	size_type& ny = dims.y;
	size_type& nz = dims.z;
	size_type& nw = dims.w;
	size_type nxy = 0, nxyz = 0;
	std::vector<T> container_host;
	pointer ptr_host = static_cast<pointer>(NULL);

	void SetDimension(
		size_type nx,
		size_type ny = 1,
		size_type nz = 1,
		size_type nw = 1
	);
	void ClearDimension() { SetDimension(0, 0, 0, 0); }

public: // Constructor & Destructor

	ArrayBase_t() = default;
	template <typename U> ArrayBase_t(const ArrayBase_t<U>& rhs)
	{ this->operator=(rhs); }
	ArrayBase_t(const ArrayBase_t<T>& rhs)
	{ this->operator=(rhs); }
	ArrayBase_t(ArrayBase_t<T>&& rhs)
	{ this->operator=(rhs); }
	explicit ArrayBase_t(
		size_type nx,
		size_type ny = 1,
		size_type nz = 1,
		size_type nw = 1
	)
	{ Resize(nx, ny, nz, nw); }
	explicit ArrayBase_t(
		const_pointer ptr,
		size_type nx,
		size_type ny = 1,
		size_type nz = 1,
		size_type nw = 1
	)
	{ Resize(nx, ny, nz, nw); copy(ptr, ptr + n); }
	~ArrayBase_t() { Clear(); ClearDimension(); }

public: // Aliasing

	void Alias(const ArrayBase_t<T>& rhs);

public: // Resizing

	void Resize(
		size_type nx,
		size_type ny = 1,
		size_type nz = 1,
		size_type nw = 1
	);
	void Resize(const dimension_t& dim)
	{ Resize(dim.x, dim.y, dim.z, dim.w); }

public: // Fill & Copy

	void fill(const_reference val);
	template <typename InIt>
	void copy(InIt begin, InIt end);

public: // Clearing

	void Clear() { container_host.clear(); }

public: // Info.

	inline bool IsAlloc() const noexcept { return !container_host.empty(); }
	inline bool IsAlias() const noexcept { return ptr_host != NULL && container_host.empty(); }
	inline dimension_t GetDims() const noexcept { return dims; }

public: // STL-Consistent Methods

	inline pointer data() noexcept { return ptr_host; }
	inline const_pointer data() const noexcept { return ptr_host; }
	inline iterator begin() noexcept { return container_host.begin(); }
	inline const_iterator begin() const noexcept { return container_host.begin(); }
	inline iterator end() noexcept { return container_host.end(); }
	inline const_iterator end() const noexcept { return container_host.end(); }
	inline reference front() noexcept { return container_host.front(); }
	inline const_reference front() const noexcept { return container_host.front(); }
	inline reference back() noexcept { return container_host.back(); }
	inline const_reference back() const noexcept { return container_host.back(); }
	inline size_type size() const noexcept { return container_host.size(); }
	
public: // Assignment Operations

	inline ArrayBase_t<T>& operator=(const_reference val);
	inline ArrayBase_t<T>& operator=(const_pointer ptr);
	template <typename U> inline ArrayBase_t<T>& operator=(const ArrayBase_t<U>& rhs);
	inline ArrayBase_t<T>& operator=(const ArrayBase_t<T>& rhs);
	inline ArrayBase_t<T>& operator=(ArrayBase_t<T>&& rhs);

public: // Arithmetic Operations

	inline ArrayBase_t<T>& operator+=(const_reference val);
	template <typename U> inline ArrayBase_t<T>& operator+=(const ArrayBase_t<U>& rhs);
	inline ArrayBase_t<T>& operator+=(const ArrayBase_t<T>& rhs);

	inline ArrayBase_t<T>& operator-=(const_reference val);
	template <typename U> inline ArrayBase_t<T>& operator-=(const ArrayBase_t<U>& rhs);
	inline ArrayBase_t<T>& operator-=(const ArrayBase_t<T>& rhs);

	inline ArrayBase_t<T>& operator*=(const_reference val);
	template <typename U> inline ArrayBase_t<T>& operator*=(const ArrayBase_t<U>& rhs);
	inline ArrayBase_t<T>& operator*=(const ArrayBase_t<T>& rhs);

	inline ArrayBase_t<T>& operator/=(const_reference val);
	template <typename U> inline ArrayBase_t<T>& operator/=(const ArrayBase_t<U>& rhs);
	inline ArrayBase_t<T>& operator/=(const ArrayBase_t<T>& rhs);

public: // Indexing Operations

	inline reference operator[] (index_type i) noexcept { return container_host[i]; }
	inline const_reference operator[] (index_type i) const noexcept { return container_host[i]; }

	inline reference operator() (
		index_type ix, 
		index_type iy = 0, 
		index_type iz = 0, 
		index_type iw = 0
	) noexcept
	{
		return container_host[iw * nxyz + iz * nxy + iy * nx + ix];
	}

	inline const_reference operator() (
		index_type ix,
		index_type iy = 0,
		index_type iz = 0,
		index_type iw = 0
		) const noexcept
	{
		return container_host[iw * nxyz + iz * nxy + iy * nx + ix];
	}

};

// Recipe for Constructor

template <typename T>
void ArrayBase_t<T>::SetDimension(
	size_type nx,
	size_type ny,
	size_type nz,
	size_type nw
)
{
	this->nx = nx;
	this->ny = ny;
	this->nz = nz;
	this->nw = nw;
	this->nxy = nx * ny;
	this->nxyz = nx * ny * nz;
	this->n = nx * ny * nz * nw;
}

template <typename T>
void ArrayBase_t<T>::Resize(
	size_type nx,
	size_type ny,
	size_type nz,
	size_type nw
)
{
	SetDimension(nx, ny, nz, nw);
	container_host.resize(n);
	ptr_host = container_host.data();
}

template <typename T>
void ArrayBase_t<T>::fill(const_reference val)
{
	std::fill(std::execution::par, begin(), end(), val);
}

template <typename T> template <typename InIt>
void ArrayBase_t<T>::copy(InIt begin, InIt end)
{
	std::copy(std::execution::par, begin, end, this->begin());
}


template <typename T> inline
ArrayBase_t<T>& ArrayBase_t<T>::operator=(const_reference val)
{
	this->fill(val);
	return *this;
}

template <typename T> inline
ArrayBase_t<T>& ArrayBase_t<T>::operator=(const_pointer ptr)
{
	this->copy(ptr, ptr + size());
	return *this;
}

template <typename T> template <typename U> inline
ArrayBase_t<T>& ArrayBase_t<T>::operator=(const ArrayBase_t<U>& rhs)
{
	Resize(rhs.GetDims());
#pragma omp parallel for schedule(guided)
	for (index_type i = 0; i < size(); ++i) container_host[i] = static_cast<T>(rhs[i]);
	return *this;
}

template <typename T> inline
ArrayBase_t<T>& ArrayBase_t<T>::operator=(const ArrayBase_t<T>& rhs)
{
	Resize(rhs.GetDims());
	this->copy(rhs.begin(), rhs.end());
	return *this;
}

template <typename T> inline
ArrayBase_t<T>& ArrayBase_t<T>::operator=(ArrayBase_t<T>&& rhs)
{
	Alias(rhs);
	rhs.Clear();
	rhs.ClearDimension();
	return *this;
}


template <typename T>
using is_device_t = typename std::enable_if_t<std::is_pod_v<T>>;
template <typename T> 
using is_not_device_t = typename std::enable_if_t<!std::is_pod_v<T>>;

} // Anonymous Namespace

// Partitioned Template Class Declaration

template <typename T, typename V = void> class Array_t;

// Host-Device Both Side

template <typename T>
class Array_t<T, is_device_t<T>> : public ArrayBase_t<T>
{
private:

	thrust::device_vector<T> container_device;
	pointer ptr_device = static_cast<pointer>(NULL);

public:

	using MyBase = ArrayBase_t<T>;

public: // Constructor & Destructor

	Array_t() : MyBase() {}
	template <typename U> Array_t(const Array_t<U>& rhs) : MyBase(rhs) {}
	Array_t(const Array_t<T>& rhs) : MyBase(rhs) {}
	Array_t(Array_t<T>&& rhs) : MyBase(rhs) {}
	explicit Array_t(
		size_type nx,
		size_type ny = 1,
		size_type nz = 1,
		size_type nw = 1
	) : MyBase(nx, ny, nz, nw) {}
	explicit Array_t(
		const_pointer ptr,
		size_type nx,
		size_type ny = 1,
		size_type nz = 1,
		size_type nw = 1
	) : MyBase(ptr, nx, ny, nz, nw) {}

private: // Resizing

	bool _check_dims(
		size_type nx,
		size_type ny = 1,
		size_type nz = 1,
		size_type nw = 1
	) const;

public: 

	void ResizeDevice();
	void ResizeDevice(
		size_type nx,
		size_type ny = 1,
		size_type nz = 1,
		size_type nw = 1
	);

public: // Info.

	inline bool IsDeviceAlloc() const noexcept { return !container_device.empty(); }
	inline bool IsDeviceAlias() const noexcept { return ptr_device != NULL && container_device.empty(); }

public: // Assignment Operations

	inline Array_t<T>& operator=(const_reference val)
	{
		MyBase::operator=(val); return *this;
	}
	inline Array_t<T>& operator=(const_pointer ptr)
	{
		MyBase::operator=(ptr); return *this;
	}
	template <typename U> inline Array_t<T>& operator=(const Array_t<U>& rhs)
	{
		MyBase::operator=(rhs); return *this;
	}
	inline Array_t<T>& operator=(const Array_t<T>& rhs)
	{
		MyBase::operator=(rhs); return *this;
	}
	inline Array_t<T>& operator=(Array_t<T>&& rhs)
	{
		MyBase::operator=(rhs); return *this;
	}
}; // Device Array_t

// Host Side

template <typename T>
class Array_t<T, is_not_device_t<T>> : public ArrayBase_t<T>
{
private:

	using MyBase = ArrayBase_t<T>;

public: // Constructor & Destructor

	Array_t() : MyBase() {}
	template <typename U> Array_t(const Array_t<U>& rhs) : MyBase(rhs) {}
	Array_t(const Array_t<T>& rhs) : MyBase(rhs) {}
	Array_t(Array_t<T>&& rhs) : MyBase(rhs) {}
	explicit Array_t(
		size_type nx,
		size_type ny = 1,
		size_type nz = 1,
		size_type nw = 1
	) : MyBase(nx, ny, nz, nw) {}
	explicit Array_t(
		const_pointer ptr,
		size_type nx,
		size_type ny = 1,
		size_type nz = 1,
		size_type nw = 1
	) : MyBase(ptr, nx, ny, nz, nw) {}

public: // Assignment Operations

	inline Array_t<T>& operator=(const_reference val) 
	{ 
		MyBase::operator=(val); return *this; 
	}
	inline Array_t<T>& operator=(const_pointer ptr) 
	{ 
		MyBase::operator=(ptr); return *this; 
	}
	template <typename U> inline Array_t<T>& operator=(const Array_t<U>& rhs) 
	{
		MyBase::operator=(rhs); return *this;
	}
	inline Array_t<T>& operator=(const Array_t<T>& rhs) 
	{ 
		MyBase::operator=(rhs); return *this; 
	}
	inline Array_t<T>& operator=(Array_t<T>&& rhs) 
	{ 
		MyBase::operator=(val); return *this; 
	}
};

}