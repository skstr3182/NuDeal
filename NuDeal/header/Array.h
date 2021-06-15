#pragma once
#include "Defines.h"
#include "CUDAExcept.h"
#include "CUDATypeTraits.h"

namespace LinPack
{

namespace _Detail // Namesapce Detail
{

template <typename T, unsigned N>
class _Static_array
{
public:
	T _Elems[N] = { 0, };

	_Static_array() = default;

	T& front() noexcept { return *_Elems; }
	const T& front() const noexcept { return *_Elems; }
	T& back() noexcept { return *(_Elems + N); }
	const T& back() const noexcept { return *(_Elems + N); }
	T *begin() noexcept { return _Elems; }
	const T* begin() const noexcept { return _Elems; }
	T *end() noexcept { return _Elems + N; }
	const T* end() const noexcept { return _Elems + N; }
	__inline__ __host__ __device__ T& operator[] (size_t idx) noexcept
	{
		return _Elems[idx];
	}
	__inline__ __host__ __device__ const T& operator[] (size_t idx) const noexcept
	{
		return _Elems[idx];
	}
};

template <unsigned N, typename... Ts>
using _call_if_t = typename std::enable_if_t<
	std::conjunction_v<std::is_integral<Ts>...> && sizeof...(Ts) <= N && N>;

template <typename T>
using _not_portable_t = typename std::enable_if_t<!CUDATypeTraits::is_portable_v<T>>;

template <typename T>
using _portable_t = typename std::enable_if_t<CUDATypeTraits::is_portable_v<T>>;

template <typename T>
static constexpr auto _cuda_allocate = [](size_t sz) {
	T *ptr; cudaCheckError( cudaMalloc(&ptr, sz * sizeof(T)) ); return ptr;
};

template <typename T>
struct _cuda_deallocate 
{ 
	void operator() (T* ptr) 
	{ 
		cudaCheckError( cudaFree(ptr) );
	} 
};

template <typename T, unsigned int N>
class _ArrayBase_t
{
public:

	using value_type = T;
	using pointer = T * ;
	using const_pointer = const T*;
	using reference = T & ;
	using const_reference = const T&;
	using size_type = size_t;
	using difference_type = std::ptrdiff_t;
	using iterator = pointer;
	using const_iterator = const_pointer;
	using index_type = long long;
	using array_type = T[];

protected:

	using Stride_t = _Detail::_Static_array<size_t, N>;

	// Helper Template
	template <typename... Ts>
	using call_if_t = _Detail::_call_if_t<N, Ts...>;


protected:

	// Member Variables
	Stride_t stride = { 0, };
	unsigned ndim = 1;
	std::unique_ptr<array_type> unique;
	pointer entry = NULL;

protected:

	void _move(_ArrayBase_t<T, N>&& rhs)
	{
		ndim = std::move(rhs.ndim);
		stride = std::move(rhs.stride);
		unique = std::move(rhs.unique);
		entry = std::move(rhs.entry);
	}

	void _clear()
	{
		ndim = 1;
		std::fill(stride.begin(), stride.end(), 0);
		unique.reset();
		entry = NULL;
	}

	void _create()
	{
		unique.reset(size() == 0 ? NULL : new T[size()]);
		entry = unique.get();
		std::fill(std::execution::par, begin(), end(), T{});
	}

	void _create(unsigned ndim, const Stride_t& stride)
	{
		this->ndim = ndim;
		this->stride = stride;
		_create();
	}

public: // Constructor

	_ArrayBase_t() = default;

	_ArrayBase_t(const _ArrayBase_t<T, N>& rhs) :
		ndim{ rhs.ndim },
		stride{ rhs.stride },
		unique{ rhs.empty() ? NULL : new T[size()] },
		entry{ unique.get() }
	{ 
		std::copy(std::execution::par, rhs.begin(), rhs.end(), begin()); 
	}

	_ArrayBase_t(_ArrayBase_t<T, N>&& rhs) noexcept :
		ndim{ std::move(rhs.ndim) },
		stride{ std::move(rhs.stride) },
		unique{ std::move(rhs.unique) },
		entry{ std::move(rhs.entry) }
	{ 
		rhs._clear(); 
	}

	// Explicit
	template <typename... Ts, typename = call_if_t<size_type, Ts...>>
	explicit _ArrayBase_t(size_type first, Ts... pack) :
		ndim{ sizeof...(Ts) + 1 },
		stride{ first, }
	{
		int i = 1; (..., (stride[i++] = stride[i - 1] * pack));
		_create();
	}

	// For Const Alias
	template <typename... Ts, typename = call_if_t<size_type, Ts...>>
	explicit _ArrayBase_t(const_pointer ptr, size_type first, Ts... pack) :
		_ArrayBase_t{ first, pack... }
	{
		std::copy(std::execution::par, ptr, ptr + size(), begin());
	}

public: // Assignment

	_ArrayBase_t<T, N>& operator=(const _ArrayBase_t<T, N>& rhs)
	{
		if (empty() && !rhs.empty()) _create(rhs.ndim, rhs.stride);
		std::copy(std::execution::par, rhs.begin(), rhs.end(), begin());
		return *this;
	}

	_ArrayBase_t<T, N>& operator=(_ArrayBase_t<T, N>&& rhs)
	{
		_move(std::move(rhs)); 
		rhs._clear(); 
		return *this;
	}

public:

	pointer data() noexcept { return entry; }
	const_pointer data() const noexcept { return entry; }
	iterator begin() noexcept { return entry; }
	const_iterator begin() const noexcept { return entry; }
	iterator end() noexcept { return begin() + size(); }
	const_iterator end() const noexcept { return begin() + size(); }
	reference front() noexcept { return entry[0]; }
	const_reference front() const noexcept { return entry[0]; }
	reference back() noexcept { return entry[size() - 1]; }
	const_reference back() const noexcept { return entry[size() - 1]; }

	constexpr bool empty() const noexcept { return !entry; }
	constexpr size_type size() const noexcept { return empty() ? 0 : stride[ndim - 1]; }

	template <unsigned Dim>
	__host__ __device__
	constexpr size_type Rank() const noexcept
	{
		static_assert(Dim <= N && Dim, "Dimension Overflow");
		if constexpr (Dim > 1)
			return stride[Dim - 1] / stride[Dim - 2];
		return stride[0];
	}

	template <unsigned Dim>
	__host__ __device__
	constexpr size_type Stride() const noexcept
	{
		static_assert(Dim <= N && Dim, "Dimension Overflow");
		if constexpr (Dim > 1)
			return stride[Dim - 2];
		return 1;
	}

public: // Resize

	template <typename... Ts, typename = call_if_t<size_type, Ts...>>
	void Create(size_type first, Ts... pack)
	{
		_move(_ArrayBase_t<T, N>(first, pack...));
	}

public: // Copy

	void Copy(const_pointer begin, const_pointer end = NULL)
	{
		size_type sz = end ? end - begin : size();
		std::copy(std::execution::par, begin, begin + sz, begin());
	}

public: // Fill

	void Fill(const_reference val)
	{
		std::fill(std::execution::par, begin(), end(), val);
	}

public: // Reshape

	template <typename... Ts, typename = call_if_t<size_type, Ts...>>
	void Reshape(size_type first, Ts... pack)
	{
		if (empty()) return;
#ifdef _DEBUG
		auto _n = size();
#endif
		ndim = 1 + sizeof...(pack);
		stride.front() = first;
		int i = 1; (..., (stride[i++] = stride[i - 1] * pack));
#ifdef _DEBUG
		assert(_n == size());
#endif
	}

public: // Clear

	void Destroy() 
	{
		unique.reset();
		entry = NULL;
	}

public: // Indexing Operator

	template <typename... Ts, typename = call_if_t<index_type, Ts...>>
	inline reference operator() (index_type idx, Ts... _seq) noexcept
	{
		if constexpr (sizeof...(Ts) > 0) {
			int i = 0; (..., (idx += _seq * stride[i++]));
		}
		return entry[idx];
	}

	template <typename... Ts, typename = call_if_t<index_type, Ts...>>
	inline const_reference operator() (index_type idx, Ts... _seq) const noexcept
	{
		if constexpr (sizeof...(Ts) > 0) {
			int i = 0; (..., (idx += _seq * stride[i++]));
		}
		return entry[idx];
	}
};

} // Namespace Detail

template <typename, unsigned = 4, typename = void> class Array_t;

// Host Array
template <typename T, unsigned N>
class Array_t<T, N, _Detail::_not_portable_t<T>> : public _Detail::_ArrayBase_t<T, N>
{
private:
		
	using HostSide = _Detail::_ArrayBase_t<T, N>;

public: // Constructor && Destructor

	using HostSide::HostSide;

public: // Assignment
	
	using HostSide::operator=;

}; // Host Array

// Device Array
template <typename T> using is_portable_t = _Detail::_portable_t<T>;

template <typename T, unsigned N>
class Array_t<T, N, is_portable_t<T>> : public _Detail::_ArrayBase_t<T, N>
{
private:

	using HostSide = _Detail::_ArrayBase_t<T, N>;

public:
	
	using reference = HostSide::reference;
	using const_reference = HostSide::const_reference;
	using size_type = HostSide::size_type;
	using index_type = HostSide::index_type;
	using pointer = HostSide::pointer;
	using const_pointer = HostSide::const_pointer;
	using iterator = HostSide::iterator;
	using const_iterator = HostSide::const_iterator;
	using array_type = HostSide::array_type;

private:

	static constexpr auto cuda_allocate = _Detail::_cuda_allocate<T>;
	using cuda_deallocate = _Detail::_cuda_deallocate<T>;

private:

	std::unique_ptr<array_type, cuda_deallocate> unique;
	pointer entry = NULL;

private:

	void _move(Array_t<T, N>&& rhs)
	{
		unique = std::move(rhs.unique);
		entry = std::move(rhs.entry);
	}

	void _clear()
	{
		unique.reset();
		entry = NULL;
	}

	void _cudamalloc()
	{
		unique.reset(size() == 0 ? NULL : cuda_allocate(size()));
		entry = unique.get();
		cudaCheckError( cudaMemset(entry, 0x00, device_size() * sizeof(T)) );
	}

	void _cudamalloc(unsigned ndim, const Stride_t& stride)
	{
		this->ndim = ndim;
		this->stride = stride;
		_cudamalloc();
	}

public: // Constructor & Destructor

	Array_t() = default;

	Array_t(const Array_t<T, N>& rhs) : 
		HostSide{ rhs },
		unique{ rhs.device_empty() ? NULL : cuda_allocate(size()) },
		entry{ unique.get() }
	{
		cudaCheckError(
			cudaMemcpy(device_data(), rhs.device_data(), device_size() * sizeof(T),
				cudaMemcpyDeviceToDevice)
		);
	}

	Array_t(Array_t<T, N>&& rhs) noexcept :
		HostSide{ std::move(rhs) },
		unique{ std::move(rhs.unique) },
		entry{ std::move(rhs.entry) }
	{ 
		rhs._clear(); 
	}

	// Explicit
	template <typename... Ts, typename = call_if_t<size_type, Ts...>>
	explicit Array_t(size_type first, Ts... pack) :
		HostSide{ first, pack... }
	{}

	template <typename... Ts, typename = call_if_t<size_type, Ts...>>
	explicit Array_t(const_pointer ptr, size_type first, Ts... pack) :
		HostSide{ ptr, first, pack... }
	{}

public: // Assignment

	Array_t<T, N>& operator=(const Array_t<T, N>& rhs)
	{ 
		HostSide::operator=(rhs);
		if (device_empty() && !rhs.device_empty()) _cudamalloc(rhs.ndim, rhs.stride);
		cudaCheckError(
			cudaMemcpy(device_data(), rhs.device_data(), device_size() * sizeof(T), cudaMemcpyDeviceToDevice)
		);
		return *this;
	}

	Array_t<T, N>& operator=(Array_t<T, N>&& rhs)
	{ 
		HostSide::_move(std::move(rhs));
		_move(std::move(rhs)); 
		rhs.HostSide::_clear();
		rhs._clear(); 
		return *this;
	}

public: // Status

	__host__ __device__ constexpr bool device_empty() const noexcept 
	{ 
		return !entry; 
	}
	__host__ __device__ pointer device_data() noexcept 
	{ 
		return entry; 
	}
	__host__ __device__ const_pointer device_data() const noexcept 
	{ 
		return entry; 
	}
	__host__ __device__ constexpr size_type device_size() const noexcept
	{
		return device_empty() ? 0 : size();
	}

public: // Resize

	template <typename... Ts, typename = call_if_t<size_type, Ts...>>
	void Create(size_type first, Ts... pack);

	template <typename... Ts, typename = call_if_t<size_type, Ts...>>
	void CreateDevice(size_type first, Ts... pack);

public: // Copy

	void CopyH(const Array_t<T, N>& rhs);
	void CopyD(const Array_t<T, N>& rhs);
	void CopyHtoD();
	void CopyDtoH();
	void CopyHtoD(const_pointer begin, const_pointer end = NULL);
	void CopyDtoH(const_pointer begin, const_pointer end = NULL);
	void CopyHtoH(const_pointer begin, const_pointer end = NULL);
	void CopyDtoD(const_pointer begin, const_pointer end = NULL);

public: // Fill

	void FillDevice(const_reference val);

public: // Reshape

	template <typename... Ts, typename = call_if_t<size_type, Ts...>>
	void Reshape(size_type first, Ts... pack);

public: // Clear

	void Destroy()
	{
		HostSide::Destroy();
	}

	void DestroyDevice()
	{
		unique.reset();
		entry = NULL;
	}

public: // Indexing

	template <typename... Ts, typename = call_if_t<index_type, Ts...>>
	__inline__ __device__ __host__
	reference operator()(index_type idx, Ts... _seq) noexcept
	{
#ifdef __CUDA_ARCH__
		if constexpr (sizeof...(Ts) > 0) {
			int i = 0; (..., (idx += stride[i++] * _seq));
		}
		return entry[idx];
#else
		return HostSide::operator()(idx, _seq...);
#endif
	}

	template <typename... Ts, typename = call_if_t<index_type, Ts...>>
	__inline__ __device__ __host__
	const_reference operator()(index_type idx, Ts... _seq) const noexcept
	{
#ifdef __CUDA_ARCH__
		if constexpr (sizeof...(Ts) > 0) {
			int i = 0; (..., (idx += stride[i++] * _seq));
		}
		return entry[idx];
#else
		return HostSide::operator()(idx, _seq...);
#endif
	}

}; // Device Array

}