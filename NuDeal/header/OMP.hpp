#pragma once

#if defined _DEBUG
#define __PRAGMA__(...)
#else
#if defined _MSC_VER
#define __PRAGMA__(...) __pragma(__VA_ARGS__)
#else
#define __PRAGMA__(...) _Pragma(#__VA_ARGS__)
#endif
#endif

#define OMP(...) __PRAGMA__(omp __VA_ARGS__)
#define OMP_FOR(...) OMP(parallel for schedule(guided) __VA_ARGS__)
#define OMP_STATIC OMP(for schedule(static))
#define OMP_DYNAMIC OMP(for schedule(dynamic))
#define OMP_GUIDED OMP(for schedule(guided))