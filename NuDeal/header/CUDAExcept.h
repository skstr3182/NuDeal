#pragma once
#include "Defines.h"

// CUDA Error Checke Macros

#define cudaCheckError(error) \
if (error != cudaError::cudaSuccess) { \
	cout << cudaGetErrorString(error) << endl; \
	cout << "In file" << __FILE__ \
		<< " line " << __LINE__ \
		<< " func " << __func__ << endl; \
	exit(EXIT_FAILURE); \
}

#define cudaCheckErrorBLAS(error) \
if (error != cudaError::cudaSuccess) { \
	cout << "cuBLAS error code : " << error << endl; \
	cout << "In file " << __FILE__ \
		<< " line " << __LINE__ << \
		<< " func " << __func__ << endl; \
	exit(EXIT_FAILURE); \
}

#define cudaCheckErrorSPARSE(error) \
if (error != cudaErrorSuccess) { \
	cout << "cuSPARSE error code : " << error << endl; \
	cout << "In file " << __FILE__ \
		<< " line " << __LINE__ << \
		<< " func " << __func__ << endl; \
	exit(EXIT_FAILURE); \
}