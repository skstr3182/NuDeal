#pragma once
#include <cuda_runtime.h>
#include <type_traits>

namespace CUDATypeTraits
{

#define _COMMIT_CUDA_INTRINSIC_VECTOR_(STRUCT, BASE, DERIVED) \
template <> \
struct STRUCT<BASE> \
{ \
	using _1 = DERIVED##1; \
	using _2 = DERIVED##2; \
	using _3 = DERIVED##3; \
	using _4 = DERIVED##4; \
};

#define _INSTANTIATE_CUDA_INTRINSIC_VECTOR_(STRUCT, BASE, TYPE_CONTAINER) \
template <> struct STRUCT<typename TYPE_CONTAINER<BASE>::_1> : true_type{}; \
template <> struct STRUCT<typename TYPE_CONTAINER<BASE>::_2> : true_type{}; \
template <> struct STRUCT<typename TYPE_CONTAINER<BASE>::_3> : true_type{}; \
template <> struct STRUCT<typename TYPE_CONTAINER<BASE>::_4> : true_type{};

template <typename T> struct cuda_vector_collect {};
_COMMIT_CUDA_INTRINSIC_VECTOR_(cuda_vector_collect, char,               char)
_COMMIT_CUDA_INTRINSIC_VECTOR_(cuda_vector_collect, unsigned char,      uchar)
_COMMIT_CUDA_INTRINSIC_VECTOR_(cuda_vector_collect, short,              short)
_COMMIT_CUDA_INTRINSIC_VECTOR_(cuda_vector_collect, unsigned short,     ushort)
_COMMIT_CUDA_INTRINSIC_VECTOR_(cuda_vector_collect, int,                int)
_COMMIT_CUDA_INTRINSIC_VECTOR_(cuda_vector_collect, unsigned int,       uint)
_COMMIT_CUDA_INTRINSIC_VECTOR_(cuda_vector_collect, long,	              long)
_COMMIT_CUDA_INTRINSIC_VECTOR_(cuda_vector_collect, unsigned long,      ulong)
_COMMIT_CUDA_INTRINSIC_VECTOR_(cuda_vector_collect, long long,          longlong)
_COMMIT_CUDA_INTRINSIC_VECTOR_(cuda_vector_collect, unsigned long long, ulonglong)
_COMMIT_CUDA_INTRINSIC_VECTOR_(cuda_vector_collect, float,              float)
_COMMIT_CUDA_INTRINSIC_VECTOR_(cuda_vector_collect, double,             double)

template <typename T> struct is_intrinsic   : false_type {};
_INSTANTIATE_CUDA_INTRINSIC_VECTOR_(is_intrinsic, char,               cuda_vector_collect)
_INSTANTIATE_CUDA_INTRINSIC_VECTOR_(is_intrinsic, unsigned char,      cuda_vector_collect)
_INSTANTIATE_CUDA_INTRINSIC_VECTOR_(is_intrinsic, short,              cuda_vector_collect)
_INSTANTIATE_CUDA_INTRINSIC_VECTOR_(is_intrinsic, unsigned short,     cuda_vector_collect)
_INSTANTIATE_CUDA_INTRINSIC_VECTOR_(is_intrinsic, int,                cuda_vector_collect)
_INSTANTIATE_CUDA_INTRINSIC_VECTOR_(is_intrinsic, unsigned int,       cuda_vector_collect)
_INSTANTIATE_CUDA_INTRINSIC_VECTOR_(is_intrinsic, long,	              cuda_vector_collect)
_INSTANTIATE_CUDA_INTRINSIC_VECTOR_(is_intrinsic, unsigned long,      cuda_vector_collect)
_INSTANTIATE_CUDA_INTRINSIC_VECTOR_(is_intrinsic, long long,          cuda_vector_collect)
_INSTANTIATE_CUDA_INTRINSIC_VECTOR_(is_intrinsic, unsigned long long, cuda_vector_collect)
_INSTANTIATE_CUDA_INTRINSIC_VECTOR_(is_intrinsic, float,              cuda_vector_collect)
_INSTANTIATE_CUDA_INTRINSIC_VECTOR_(is_intrinsic, double,             cuda_vector_collect)

template <typename T>
inline static constexpr bool is_intrinsic_v = is_intrinsic<T>::value;

template <typename T>
struct is_portable : std::bool_constant<std::is_arithmetic_v<T> || is_intrinsic_v<T>>
{};

template <typename T>
inline static constexpr bool is_portable_v = is_portable<T>::value;

#undef _COMMIT_CUDA_INTRINSIC_VECTOR_
#undef _INSTANTIATE_CUDA_INTRINSIC_VECTOR_
}