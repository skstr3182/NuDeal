#pragma once
#include "Defines.h"
#include "CUDAExcept.h"

namespace LinPack
{

namespace // Anonymous Namespace
{

struct Layout_t
{
	using size_type = size_t;
	size_type n = 0;
	size_type nx = 0, ny = 0, nz = 0, nw = 0;
	size_type nxy = 0, nxyz = 0;
	Layout_t() = default;
	Layout_t(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1)
	{
		this->operator()(nx, ny, nz, nw);
	}
	void operator() (size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1)
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
	inline bool operator==(const Layout_t& rhs)
	{
		if (this->nx != rhs.nx) return false;
		if (this->ny != rhs.ny) return false;
		if (this->nz != rhs.nz) return false;
		if (this->nw != rhs.nw) return false;
		return true;
	}
	inline bool operator!=(const Layout_t& rhs)
	{
		return !this->operator==(rhs);
	}
};

// Base Array Type
template <typename T>
class ArrayBase_t // ArrayBase_t<T>
{
public:

	using reference = T & ;
	using const_reference = const T&;
	using size_type = Layout_t::size_type;
	using index_type = long long;
	using value_type = T;
	using pointer = T * ;
	using const_pointer = const T*;
	using iterator = pointer;
	using const_iterator = const_pointer;
	using array_type = T[];

protected:

	Layout_t L;
	std::unique_ptr<array_type> unique_host;
	pointer entry_host = NULL;

protected:

	void swap(ArrayBase_t<T>& rhs)
	{
		std::swap(L, rhs.L);
		std::swap(unique_host, rhs.unique_host);
		std::swap(entry_host, rhs.entry_host);
	}

	void shallow_copy(const ArrayBase_t<T>& rhs)
	{
		L = rhs.L;
		entry_host = rhs.entry_host;
	}

public: // Constructor & Destructor

	// Default & Copy & Move
	ArrayBase_t() = default;
	ArrayBase_t(const ArrayBase_t<T>& rhs);
	ArrayBase_t(ArrayBase_t<T>&& rhs);

	// Explicit
	explicit
		ArrayBase_t(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);

public: // Assignment

	inline ArrayBase_t<T>& operator=(const ArrayBase_t<T>& rhs);
	inline ArrayBase_t<T>& operator=(ArrayBase_t<T>&& rhs);

public: // Basic Info

	inline bool IsAlloc() const noexcept { return entry_host && unique_host; }
	inline bool IsAlias() const noexcept { return entry_host && !unique_host; }
	inline bool HasHost() const noexcept { return entry_host; }

	inline pointer host_ptr() noexcept { return entry_host; }
	inline const_pointer host_ptr() const noexcept { return entry_host; }

public:

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

	void Clear()
	{
		L.clear(); unique_host.reset(); entry_host = NULL;
	}

public: // Resize

	void Resize(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);

public: // Copy

	void Copy(const ArrayBase_t<T>& rhs);
	void Copy(const_pointer begin, const_pointer end = NULL);

public: // Alias

	void Alias(ArrayBase_t<T>& rhs);
	void Alias(pointer ptr, size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);

public: // Fill

	void Fill(const_reference val);

public: // Indexing Operator

	inline reference operator[] (index_type i) noexcept
	{
		return entry_host[i];
	}

	inline const_reference operator[] (index_type i) const noexcept
	{
		return entry_host[i];
	}

	inline reference operator()
		(index_type x, index_type y = 0, index_type z = 0, index_type w = 0) noexcept
	{
		return entry_host[x + y * L.nx + z * L.nxy + w * L.nxyz];
	}

	inline const_reference operator()
		(index_type x, index_type y = 0, index_type z = 0, index_type w = 0) const noexcept
	{
		return entry_host[x + y * L.nx + z * L.nxy + w * L.nxyz];
	}

}; // ArrayBase_t<T>

template <typename T>
ArrayBase_t<T>::ArrayBase_t(const ArrayBase_t<T>& rhs)
{
	shallow_copy(rhs);
}

template <typename T>
ArrayBase_t<T>::ArrayBase_t(ArrayBase_t<T>&& rhs)
{
	swap(rhs);
}

template <typename T>
ArrayBase_t<T>::ArrayBase_t(size_type nx, size_type ny, size_type nz, size_type nw)
{
	L(nx, ny, nz, nw);
	unique_host = std::make_unique<array_type>(L.n);
	entry_host = unique_host.get();
}

template <typename T>
ArrayBase_t<T>& ArrayBase_t<T>::operator=(const ArrayBase_t<T>& rhs)
{
	Clear();
	shallow_copy(rhs);
	return *this;
}

template <typename T>
ArrayBase_t<T>& ArrayBase_t<T>::operator=(ArrayBase_t<T>&& rhs)
{
	Clear();
	swap(rhs);
	return *this;
}

template <typename T>
using is_host_t = typename std::enable_if<!std::is_pod<T>::value>::type;
template <typename T>
using is_device_t = typename std::enable_if<std::is_pod<T>::value>::type;

template <typename, typename = void> class cuda_allocator;

template <typename T>
class cuda_allocator<T, is_device_t<T>>
{
public:
	inline static constexpr auto allocate = [](size_t sz) 
	{
		T *ptr; cudaCheckError(cudaMalloc(&ptr, sz * sizeof(T))); return ptr;
	};
	inline static constexpr auto deallocate = [](T *ptr)
	{
		cudaCheckError(cudaFree(ptr));
	};
};

} // Anonymous Namespace

/* Split Array Type								 */
/* Host-Specialized : Not POD Type */
/* Host-Device : POD Type					 */
template <typename, typename = void> class Array_t;

// Host-Specialized Array Type
template <typename T>
class Array_t<T, typename is_host_t<T>> : public ArrayBase_t<T>
{
private:
	
	using MyBase = ArrayBase_t<T>;

public: // Constructor & Destructor

	// Default & Copy & Move
	Array_t() = default;
	Array_t(const Array_t<T>& rhs) : MyBase(rhs) {}
	Array_t(Array_t<T>&& rhs) : MyBase(rhs) {}

	// Explicit
	explicit 
	Array_t(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1)
		: MyBase(nx, ny, nz, nw) {}

public: // Assignment

	inline Array_t<T>& operator=(const Array_t<T>& rhs) { MyBase::operator=(rhs); return *this; }
	inline Array_t<T>& operator=(Array_t<T>&& rhs) { MyBase::operator=(rhs); return *this; }

public: // Indexing Operator

	inline reference operator[] (index_type i) noexcept
	{
		return MyBase::operator[](i);
	}

	inline const_reference operator[] (index_type i) const noexcept
	{
		return MyBase::operator[](i);
	}

	inline reference operator()
		(index_type x, index_type y = 0, index_type z = 0, index_type w = 0) noexcept
	{
		return MyBase::operator()(x, y, z, w);
	}

	inline const_reference operator()
		(index_type x, index_type y = 0, index_type z = 0, index_type w = 0) const noexcept
	{
		return MyBase::operator()(x, y, z, w);
	}
};

// Host-Device Array Type
template <typename T>
class Array_t<T, typename is_device_t<T>> : public ArrayBase_t<T>
{
private:
	
	using MyBase = ArrayBase_t<T>;
	
	inline static constexpr auto allocate = cuda_allocator<T>::allocate;
	inline static constexpr auto deallocate = cuda_allocator<T>::deallocate;

	inline static constexpr auto make_device_ptr(size_t sz)
	{
		return std::unique_ptr<array_type, std::function<void(T*)>>(allocate(sz), deallocate);
	}

private:

	Layout_t L;
	std::unique_ptr<array_type, std::function<void(T*)>> unique_device;
	pointer entry_device = NULL;

private:

	void swap(Array_t<T>& rhs)
	{
		MyBase::swap(rhs);
		std::swap(L, rhs.L);
		std::swap(entry_device, rhs.entry_device);
		std::swap(unique_device, rhs.unique_device);
	}

	void shallow_copy(const Array_t<T>& rhs)
	{
		MyBase::shallow_copy(rhs);
		entry_device = rhs.entry_device;
		L = rhs.L;
	}

public: // Constructor & Destructor

	Array_t() = default;
	Array_t(const Array_t<T>& rhs);
	Array_t(Array_t<T>&& rhs);

	// Explicit
	explicit 
	Array_t(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1)
		: MyBase(nx, ny, nz, nw) {}

public: // Assignment
	
	inline Array_t<T>& operator=(const Array_t<T>& rhs);
	inline Array_t<T>& operator=(Array_t<T>&& rhs);
	
public:

	inline bool IsDeviceAlloc() const noexcept { return entry_device && unique_device; }
	inline bool IsDeviceAlias() const noexcept { return entry_device && !unique_device; }
	inline bool HasDevice() const noexcept { return entry_device; }

	inline pointer device_ptr() noexcept { return entry_device; }
	inline const_pointer device_ptr() const noexcept { return entry_device; }

	inline size_type device_size() const noexcept { return L.n; }
	inline size_type host_size() const noexcept { return size(); }

public: // Clear

	void ClearHost()
	{
		MyBase::Clear();
	}

	void ClearDevice()
	{
		L.clear(); unique_device.reset(); entry_device = NULL;
	}

	void Clear()
	{
		MyBase::Clear(); 
		L.clear(); unique_device.reset(); entry_device = NULL;
	}

public: // Resize

	void ResizeHost(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);
	void ResizeHost();
	void ResizeDevice(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);
	void ResizeDevice();
	void Resize(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);

public: // Copy

	void Copy(const Array_t<T>& rhs);
	void Copy(const_pointer begin, const_pointer end = NULL);
	void CopyHtoD();
	void CopyHtoD(const_pointer begin, const_pointer end = NULL);
	void CopyDtoH();
	void CopyDtoH(const_pointer begin, const_pointer end = NULL);

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
		return MyBase::operator[](i);
#endif
	}

	__forceinline__ __host__ __device__
		const_reference operator[] (index_type i) const noexcept
	{
#ifdef __CUDA_ARCH__
		return entry_device[i];
#else
		return MyBase::operator[](i);
#endif
	}

	__forceinline__ __host__ __device__
		reference operator() (index_type x, index_type y = 0, index_type z = 0, index_type w = 0) noexcept
	{
#ifdef __CUDA_ARCH__
		return entry_device[x + y * L.nx + z * L.nxy + w * L.nxyz];
#else
		return MyBase::operator()(x, y, z, w);
#endif
	}

	__forceinline__ __host__ __device__
		const_reference operator() (index_type x, index_type y = 0, index_type z = 0, index_type w = 0) const noexcept
	{
#ifdef __CUDA_ARCH__
		return entry_device[x + y * L.nx + z * L.nxy + w * L.nxyz];
#else
		return MyBase::operator()(x, y, z, w);
#endif
	}

};

template <typename T>
Array_t<T, is_device_t<T>>::Array_t(const Array_t<T>& rhs)
{
	shallow_copy(rhs);
}

template <typename T>
Array_t<T, is_device_t<T>>::Array_t(Array_t<T>&& rhs)
{
	swap(rhs);
}

template <typename T>
Array_t<T>& Array_t<T, is_device_t<T>>::operator=(const Array_t<T>& rhs)
{
	Clear();
	shallow_copy(rhs);
	return *this;
}

template <typename T>
Array_t<T>& Array_t<T, is_device_t<T>>::operator=(Array_t<T>&& rhs)
{
	Clear();
	swap(rhs);
	return *this;
}

}