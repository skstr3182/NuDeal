#pragma once
#include "Defines.h"

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
	pointer Entry = static_cast<pointer>(NULL);
	pointer d_Entry = static_cast<pointer>(NULL);
	State state = State::Undefined;
	State d_state = State::Undefined;

	void SetDimension(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);
	inline void SetHostState(State s) { state = s; }
	inline void SetDeviceState(State s) { d_state = s; }

public: // Constructor & Destructor

	Array_t() {}
	template <typename U> Array_t(const Array_t<U>& rhs) { this->operator=(rhs); }
	Array_t(const Array_t<T>& rhs) { this->operator=(rhs); }
	Array_t(Array_t<T>&& rhs) { this->operator=(rhs); }
	Array_t(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1) 
	{ ResizeHost(nx, ny, nz, nw); }
	Array_t(const_pointer ptr, size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1)
	{ ResizeHost(ptr, nx, ny, nz, nw); }
	~Array_t() { Clear(); }

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
	void Resize(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1) 
	{ ResizeHost(nx, ny, nz, nw); ResizeDevice(nx, ny, nz, nw); }

public: // Clearing

	void ClearHost();
	void ClearDevice();	
	void Clear() { ClearHost(); ClearDevice(); }

public: // Info.

	bool IsHostAlloc() const noexcept { return state == State::Alloc; }
	bool IsDeviceAlloc() const noexcept { return d_state == State::Alloc; }
	bool IsHostAlias() const noexcept { return state == State::Alias; }
	bool IsDeviceAlias() const noexcept { return d_state == State::Alias; }

public : // STL-Consistent Methods

	inline iterator begin() noexcept { return Entry; }
	inline const_iterator begin() const noexcept { return Entry; }
	inline iterator end() noexcept { return Entry + n; }
	inline const_iterator end() const noexcept { return Entry + n; }
	inline reference front() noexcept { return Entry[0]; }
	inline const_reference front() const noexcept { return Entry[0]; }
	inline reference back() noexcept { return Entry[n - 1]; }
	inline const_reference back() const noexcept { return Entry[n - 1]; }
	inline size_type size() const noexcept { return n; }

public : // Arithmatic Operations

	inline Array_t<T>& operator=(const_reference val);
	inline Array_t<T>& operator=(const_pointer ptr);
	template <typename U> inline Array_t<T>& operator=(const Array_t<U>& rhs);
	inline Array_t<T>& operator=(const Array_t<T>& rhs);
	inline Array_t<T>& operator=(Array_t<T>&& rhs);


public : // Indexing Operations

	__forceinline__ __host__ __device__ reference 
		operator[] (index_type i) noexcept 
	{ 
#ifdef __CUDA_ARCH__
		return d_Entry[i];
#else
		return Entry[i];
#endif
	}
	__forceinline__ __host__ __device__ const_reference 
		operator[] (index_type i) const noexcept 
	{ 
#ifdef __CUDA_ARCH__
		return d_Entry[i];
#else
		return Entry[i];
#endif
	}
	__forceinline__ __host__ __device__ reference 
		operator() (index_type ix, index_type iy = 0, index_type iz = 0, index_type iw = 0) noexcept 
	{ 
#ifdef __CUDA_ARCH__
		return d_Entry[iw * nxyz + iz * nxy + iy * nx + ix];
#else
		return Entry[iw * nxyz + iz * nxy + iy * nx + ix]; 
#endif
	}
	__forceinline__ __host__ __device__ const_reference
		operator() (index_type ix, index_type iy = 0, index_type iz = 0, index_type iw = 0) const noexcept
	{
#ifdef __CUDA_ARCH__
		return d_Entry[iw * nxyz + iz * nxy + iy * nx + ix];
#else
		return Entry[iw * nxyz + iz * nxy + iy * nx + ix];
#endif
	}
};

}