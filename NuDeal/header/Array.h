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
		Undefined = -1
	};

private:

	size_type n = 0;
	size_type nx = 0, ny = 0, nz = 0, nw = 0;
	size_type nxy = 0, nxyz = 0;
	pointer Entry = static_cast<pointer>(NULL);
	pointer d_Entry = static_cast<pointer>(NULL);
	State hostState = State::Undefined;
	State devState = State::Undefined;

	void SetDimension(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);

public: // Constructor & Destructor

	Array_t() {}
	template <typename U> Array_t(const Array_t<U>& rhs);
	Array_t(const Array_t<T>& rhs);
	Array_t(size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1) 
	{ ResizeHost(nx, ny, nz, nw); }
	Array_t(const_pointer ptr, size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1)
	{ ResizeHost(ptr, nx, ny, nz, nw); }
	~Array_t() { Clear(); }

public: // Aliasing 

	void Alias(const Array_t<T>&);
	void AliasHost(const Array_t<T>&);
	void AliasHost(pointer ptr, size_type nx, size_type ny = 1, size_type nz = 1, size_type nw = 1);
	void AliasDevice(const Array_t<T>&);
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

	bool IsHostAlloc() const noexcept { return hostState == State::Alloc; }
	bool IsDeviceAlloc() const noexcept { return devState == State::Alloc; }
	bool IsHostAlias() const noexcept { return hostState == State::Alias; }
	bool IsDeviceAlias() const noexcept { return devState == State::Alias; }

public : // STL-Consistent Methods

	iterator begin() noexcept { return Entry; }
	const_iterator begin() const noexcept { return Entry; }
	iterator end() noexcept return { Entry + n; }
	const_iterator end() const noexcept { return Entry + n; }
	reference front() noexcept { return Entry[0]; }
	const_reference front() const noexcept { return Entry[0]; }
	reference back() noexcept { return Entry[n - 1]; }
	const_reference back() const noexcept { return Entry[n - 1]; }

public : // Operator

	reference operator[] (index_type i) noexcept { return Entry[i]; }
	const_reference operator[] (index_type i) const noexcept { return Entry[i]; }


};

}