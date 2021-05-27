#pragma once
#include "Defines.h"
#include <thrust/device_vector.h>

namespace LinPack
{

template <typename T>
class Array_t
{
template <typename U> friend class Array_t;
public:

	using reference       = T&;
	using const_reference = const T&;
	using size_type       = size_t;
	using index_type			= long long;
	using value_type      = T;
	using pointer         = T*;
	using const_pointer   = const T*;
	using iterator        = pointer;
	using const_iterator  = const_pointer;

	enum class State
	{
		Alloc,
		Alias,
		Undefined = -1,
	};

private:

	size_type n = 0;
	size_type nx = 0, ny = 0, nz = 0, nw = 0;
	size_type nxy = 0, nxyz = 0;
	std::vector<T> container_host;
	thrust::device_vector<T> container_device;
	pointer ptr_host = static_cast<pointer>(NULL);
	pointer ptr_device = static_cast<pointer>(NULL);

	void SetDimension(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);

public: // Constructor & Destructor

	Array_t() = default;
	template <typename U> Array_t(const Array_t<U>& rhs) 
	{ this->operator=(rhs); }
	Array_t(const Array_t<T>& rhs) 
	{ this->operator=(rhs); }
	Array_t(Array_t<T>&& rhs) 
	{ this->operator=(rhs); }
	explicit Array_t
	(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1) 
	{ ResizeHost(nx, ny, nz, nw); }
	explicit Array_t
	(const_pointer ptr, size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1)
	{ ResizeHost(ptr, nx, ny, nz, nw); }
	~Array_t() 
	{ Clear(); }

public: // Aliasing 

	void Alias(const Array_t<T>& rhs);
	void AliasHost(const Array_t<T>& rhs);
	void AliasHost(pointer ptr, size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);
	void AliasDevice(const Array_t<T>& rhs);
	void AliasDevice(pointer ptr, size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);

public: // Resizing

	void ResizeHost(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);
	void ResizeHost(const_pointer ptr, size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);
	void ResizeDevice(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);
	void ResizeDevice(const_pointer ptr, size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);
	void ResizeDevice()
	{ ResizeDevice(nx, ny, nz, nw); }
	void Resize(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1) 
	{ ResizeHost(nx, ny, nz, nw); ResizeDevice(nx, ny, nz, nw); }

public: // Clearing

	void ClearHost();
	void ClearDevice();	
	void Clear() { ClearHost(); ClearDevice(); }

public: // Fill

	void Fill(const T& value);
	void FillDevice(const T& value);

public: // Info.

	inline bool IsHostAlloc() const noexcept 
	{ return !container_host.empty(); }
	inline bool IsDeviceAlloc() const noexcept 
	{ return !container_device.empty(); }
	inline bool IsHostAlias() const noexcept 
	{ return ptr_host != NULL && container_host.empty(); }
	inline bool IsDeviceAlias() const noexcept 
	{ return ptr_device != NULL && container_device.empty(); }
	
	inline pointer GetHostPointer() noexcept { return ptr_host; }
	inline const_pointer GetHostPointer() const noexcept { return ptr_host; }
	__forceinline__ __host__ __device__ 
	pointer GetDevicePointer() noexcept { return ptr_device; }
	__forceinline__ __host__ __device__ 
	const_pointer GetDevicePointer() const noexcept { return ptr_device; }

public : // STL-Consistent Methods

	inline iterator begin() noexcept { return ptr_host; }
	inline const_iterator begin() const noexcept { return ptr_host; }
	inline iterator end() noexcept { return ptr_host + n; }
	inline const_iterator end() const noexcept { return ptr_host + n; }
	inline reference front() noexcept { return ptr_host[0]; }
	inline const_reference front() const noexcept { return ptr_host[0]; }
	inline reference back() noexcept { return ptr_host[n - 1]; }
	inline const_reference back() const noexcept { return ptr_host[n - 1]; }
	inline size_type size() const noexcept { return n; }
	inline pointer data() noexcept { return ptr_host; }
	inline const_pointer data() const noexcept { return ptr_host; }

public : // Arithmatic Operations

	inline Array_t<T>& operator=(const_reference val);
	inline Array_t<T>& operator=(const_pointer ptr);
	template <typename U> inline Array_t<T>& operator=(const Array_t<U>& rhs);
	inline Array_t<T>& operator=(const Array_t<T>& rhs);
	inline Array_t<T>& operator=(Array_t<T>&& rhs);

	inline Array_t<T>& operator+=(const_reference val);
	template <typename U> inline Array_t<T>& operator+=(const Array_t<U>& rhs);
	inline Array_t<T>& operator+=(const Array_t<T>& rhs);

	inline Array_t<T>& operator-=(const_reference val);
	template <typename U> inline Array_t<T>& operator-=(const Array_t<U>& rhs);
	inline Array_t<T>& operator-=(const Array_t<T>& rhs);

	inline Array_t<T>& operator*=(const_reference val);
	template <typename U> inline Array_t<T>& operator*=(const Array_t<U>& rhs);
	inline Array_t<T>& operator*=(const Array_t<T>& rhs);

	inline Array_t<T>& operator/=(const_reference val);
	template <typename U> inline Array_t<T>& operator/=(const Array_t<U>& rhs);
	inline Array_t<T>& operator/=(const Array_t<T>& rhs);

public : // Indexing Operations

	__forceinline__ __host__ __device__ 
	reference operator[] (index_type i) 
	noexcept 
	{ 
#ifdef __CUDA_ARCH__
		return ptr_device[i];
#else
		return ptr_host[i];
#endif
	}
	__forceinline__ __host__ __device__ const_reference operator[] (index_type i) 
	const noexcept 
	{ 
#ifdef __CUDA_ARCH__
		return ptr_device[i];
#else
		return ptr_host[i];
#endif
	}
	__forceinline__ __host__ __device__ 
	reference operator() (index_type ix, index_type iy = 0, index_type iz = 0, index_type iw = 0) 
	noexcept 
	{ 
#ifdef __CUDA_ARCH__
		return ptr_device[iw * nxyz + iz * nxy + iy * nx + ix];
#else
		return ptr_host[iw * nxyz + iz * nxy + iy * nx + ix]; 
#endif
	}
	__forceinline__ __host__ __device__ 
	const_reference operator() (index_type ix, index_type iy = 0, index_type iz = 0, index_type iw = 0) 
	const noexcept
	{
#ifdef __CUDA_ARCH__
		return ptr_device[iw * nxyz + iz * nxy + iy * nx + ix];
#else
		return ptr_host[iw * nxyz + iz * nxy + iy * nx + ix];
#endif
	}
};

}