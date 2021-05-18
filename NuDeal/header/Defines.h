#pragma once

// Standard C++ Headers

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <stack>
#include <queue>
#include <vector>
#include <array>
#include <initializer_list>
#include <map>
#include <deque>
#include <set>
#include <numeric>
#include <chrono>
#include <cfloat>
#include <cassert>
#include <thread>
#include <regex>
#include <type_traits>
#include <unordered_set>
#include <unordered_map>

/// C++17 Required

#ifndef __CUDACC__

#if defined(_MSC_VER) && _MSVC_LANG > 201402L
#include <filesystem>
#elif defined(__GNUC__) && __cplusplus > 201402L
#include <filesystem>
#else
#include <experimental/filesystem>
#endif

#if defined(_MSC_VER) && _MSVC_LANG > 201402L
namespace filesys = std::filesystem;
#elif defined(__GNUC__) && __cplusplus > 201402L
namespace filesys = std::filesystem;
#else
namespace filesys = std::experimental::filesystem;
#endif

#endif

// OS dependent Headers

#ifdef _WIN64
#define NOMINMAX
#include <Windows.h>
#endif

#ifdef __linux__
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <limits.h>
#endif

#ifdef __GNUC__
#include <execinfo.h>
#endif

// CUDA Headers

#include "cuda.h"
#include "cuda_runtime.h"
#ifdef __CUDACC__
#include "device_launch_parameters.h"
#endif
#include "cublas_v2.h"
#include "cusparse_v2.h"

#ifdef _DEBUG
#define CUDA_DEBUG
#endif

// MPI Header

#include "mpi.h"

// OpenMP Header

#include "omp.h"

// HDF5 Header

#include "hdf5.h"
#include "hdf5_hl.h"

// Type Definitions

#ifdef FP64
	
using real = double;
using real2 = double2;
using real3 = double3;
using real4 = double4;

#define make_real2 make_double2;
#define make_real3 make_double3;
#define make_real4 make_double4;

#else

using real = float;
using real2 = float2;
using real3 = float3;
using real4 = float4;

#define make_real2 make_float2;
#define make_real3 make_float3;
#define make_real4 make_float4;

#endif