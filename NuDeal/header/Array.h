#pragma once
#include "Defines.h"
#include "CUDAExcept.h"

namespace LinPack
{

template <typename T, typename = void>
class ArrayBase_t;



template <typename T>
class Array_t // Array_t<T>
{
public:

	using reference = T&;
	using const_reference = const T&;
	using size_type = size_t;
	using index_type = long long;
	using value_type = T;
	using pointer = T*;
	using const_pointer = const T*;
	using iterator = pointer;
	using const_iterator = const_pointer;
	using array_type = T[];

public:
	
	struct Layout_t
	{
		size_type n = 0;
		size_type nx = 0, ny = 0, nz = 0, nw = 0;
		size_type nxy = 0, nxyz = 0;
		void operator() (
			size_type nx,
			size_type ny = 1,
			size_type nz = 1,
			size_type nw = 1
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
		void clear() { this->operator()(0, 0, 0, 0); }
		bool check(
			size_type nx,
			size_type ny = 1,
			size_type nz = 1,
			size_type nw = 1
		)
		{
			if (this->nx != nx) return false;
			if (this->ny != ny) return false;
			if (this->nz != nz) return false;
			if (this->nw != nw) return false;
			return true;
		}
	};

private:

	class cudaDeleter_t
	{
	public:
		void operator()(T *ptr) { cudaCheckError(cudaFree(ptr)); }
	};

	static constexpr cudaDeleter_t cuda_deleter = cudaDeleter_t();
	static inline constexpr auto cuda_allocator = [](size_t n) {
		T *ptr; cudaCheckError(cudaMalloc(&ptr, n * sizeof(T))); return ptr;
	};

private:

	Layout_t L;
	std::unique_ptr<array_type> unique_host;
	std::unique_ptr<array_type, cudaDeleter_t> unique_device;
	pointer entry_host = NULL, entry_device = NULL;

private:

	void swap(Array_t<T>& rhs)
	{
		std::swap(L, rhs.L);
		std::swap(unique_host, rhs.unique_host);
		std::swap(unique_device, rhs.unique_device);
		std::swap(entry_host, rhs.entry_host);
		std::swap(entry_device, rhs.entry_device);
	}

	void shallow_copy(const Array_t<T>& rhs)
	{
		L = rhs.L;
		entry_host = rhs.entry_host;
		entry_device = rhs.entry_device;
	}

public: // Constructor & Destructor

	// Default & Copy & Move
	Array_t() = default;
	Array_t(const Array_t<T>& rhs);
	Array_t(Array_t<T>&& rhs);

	// Explicit
	explicit 
	Array_t(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);

public: // Assignment

	inline Array_t<T>& operator=(const Array_t<T>& rhs);
	inline Array_t<T>& operator=(Array_t<T>&& rhs);

public: // Basic Info

	inline bool IsHostAlloc() const noexcept { return entry_host && unique_host; }
	inline bool IsHostAlias() const noexcept { return entry_host && !unique_host; }
	inline bool IsDeviceAlloc() const noexcept { return entry_device && unique_device; }
	inline bool IsDeviceAlias() const noexcept { return entry_device && !unique_device; }
	inline bool HasHost() const noexcept { return entry_host; }
	inline bool HasDevice() const noexcept { return entry_device; }

	inline pointer host_ptr() noexcept{ return entry_host; }
	inline const_pointer host_ptr() const noexcept { return entry_host; }
	__forceinline__ __host__ __device__
	pointer dev_ptr() noexcept { return entry_device; }
	__forceinline__ __host__ __device__
	const_pointer dev_ptr() const noexcept { return entry_device; }

public: // STL-Consistent Methods

	inline pointer data() noexcept { return entry_host; }
	inline const_pointer data() const noexcept { return entry_host; }
	inline iterator begin() noexcept { return entry_host; }
	inline const_iterator begin() const noexcept { return entry_host; }
	inline iterator end() noexcept { return entry_host + L.n; }
	inline const_iterator end() const noexcept { return entry_host + L.n; }
	inline reference front() noexcept { return entry_host[0]; }
	inline const_reference front() const noexcept { return entry_host[0]; }
	inline reference back() noexcept { return entry_host[L.n - 1]; }
	inline const_reference back() const noexcept { return entry_host[L.n - 1]; }
	inline size_type size() const noexcept { return L.n; }

public: // Clear
	
	void Clear();

public: // Resize

	void Resize(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);
	void ResizeDevice();
	void ResizeDevice(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);

public: // Copy
	
	void CopyHtoD();
	void CopyDtoH();
	void Copy(const Array_t<T>& rhs);
	void Copy(const_pointer begin, const_pointer end = NULL);

public: // Alias

	void Alias(Array_t<T>& rhs);
	void Alias(pointer ptr, size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);

public: // Fill

	void Fill(const_reference val);

public: // Indexing Operator

	__forceinline__ __host__ __device__
		reference operator[] (index_type i) noexcept
	{
#ifdef __CUDA_ARCH__
		return entry_device[i];
#else
		return entry_host[i];
#endif
	}

	__forceinline__ __host__ __device__
		const_reference operator[] (index_type i) const noexcept
	{
#ifdef __CUDA_ARCH__
		return entry_device[i];
#else
		return entry_host[i];
#endif
	}

	__forceinline__ __host__ __device__
		reference operator()
		(index_type x, index_type y = 0, index_type z = 0, index_type w = 0) noexcept
	{
#ifdef __CUDA_ARCH__
		return entry_device[x + y * L.nx + z * L.nxy + w * L.nxyz];
#else
		return entry_host[x + y * L.nx + z * L.nxy + w * L.nxyz];
#endif
	}

	__forceinline__ __host__ __device__
		const_reference operator()
		(index_type x, index_type y = 0, index_type z = 0, index_type w = 0) const noexcept
	{
#ifdef __CUDA_ARCH__
		return entry_device[x + y * L.nx + z * L.nxy + w * L.nxyz];
#else
		return entry_host[x + y * L.nx + z * L.nxy + w * L.nxyz];
#endif
	}
}; // Array_t<T>

template <typename T>
Array_t<T>::Array_t(const Array_t<T>& rhs)
{
	shallow_copy(rhs);
}

template <typename T>
Array_t<T>::Array_t(Array_t<T>&& rhs)
{
	swap(rhs);
}

template <typename T>
Array_t<T>::Array_t(size_type nx, size_type ny, size_type nz, size_type nw)
{
	L(nx, ny, nz, nw);
	unique_host = std::make_unique<array_type>(L.n);
	entry_host = unique_host.get();
}

template <typename T>
Array_t<T>& Array_t<T>::operator=(const Array_t<T>& rhs)
{
	Clear();
	shallow_copy(rhs);
	return *this;
}

template <typename T>
Array_t<T>& Array_t<T>::operator=(Array_t<T>&& rhs)
{
	Clear();
	swap(rhs);
	return *this;
}

template <typename T>
void Array_t<T>::Clear()
{
	L.clear();
	unique_host.reset(); entry_host = NULL;
	unique_device.reset(); entry_device = NULL;
}

}