/**
 * ALGEBRA PLUGIN library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#if defined(__CUDACC__) || defined(__HIP__)
#define ALGEBRA_DEVICE __device__
#else
#define ALGEBRA_DEVICE
#endif

#if defined(__CUDACC__) || defined(__HIP__)
#define ALGEBRA_HOST __host__
#else
#define ALGEBRA_HOST
#endif

#if defined(__CUDACC__) || defined(__HIP__)
#define ALGEBRA_HOST_DEVICE __host__ __device__
#else
#define ALGEBRA_HOST_DEVICE
#endif

#if defined(__CUDACC__) || defined(__HIP__)
#define ALGEBRA_ALIGN(x) __align__(x)
#else
#define ALGEBRA_ALIGN(x) alignas(x)
#endif
