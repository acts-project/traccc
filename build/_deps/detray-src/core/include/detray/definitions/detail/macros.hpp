/**
 * DETRAY library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Define a macro that assures there is no device compilation
#if not defined(__CUDACC__) && not defined(CL_SYCL_LANGUAGE_VERSION) && \
    not defined(SYCL_LANGUAGE_VERSION) && not defined(__HIP__)
#define __NO_DEVICE__
#endif
