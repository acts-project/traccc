/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#if defined(__CUDACC__)
#define TRACCC_DEVICE __device__
#else
#define TRACCC_DEVICE
#endif

#if defined(__CUDACC__)
#define TRACCC_HOST __host__
#else
#define TRACCC_HOST
#endif

#if defined(__CUDACC__)
#define TRACCC_HOST_DEVICE __host__ __device__
#else
#define TRACCC_HOST_DEVICE
#endif
