/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#if not defined __has_builtin
#define TRACCC_ASSUME(cond)
#elif __has_builtin(__builtin_assume)
#define TRACCC_ASSUME(cond) __builtin_assume(cond)
#else
#define TRACCC_ASSUME(cond)
#endif

#if defined(__CUDACC__) || defined(__HIP__) || defined(__OPENMP) || \
    defined(__SYCL__)
#define TRACCC_PRAGMA_UNROLL _Pragma("unroll")
#else
#define TRACCC_PRAGMA_UNROLL
#endif
