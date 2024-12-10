/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/device_address_space.hpp"

// System include(s).
#include <atomic>

#if defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
#if defined(VECMEM_HAVE_SYCL_ATOMIC_REF)

// Local include(s).
#include "vecmem/memory/details/sycl_builtin_device_atomic_ref.hpp"

namespace vecmem {

/// Use @c vecmem::sycl::builtin_device_atomic_ref with modern SYCL code
template <typename T,
          device_address_space address = device_address_space::global>
using device_atomic_ref = sycl::builtin_device_atomic_ref<T, address>;

}  // namespace vecmem

#else  // defined(VECMEM_HAVE_SYCL_ATOMIC_REF)

// Local include(s).
#include "vecmem/memory/details/sycl_custom_device_atomic_ref.hpp"

namespace vecmem {

/// Use @c vecmem::sycl::custom_device_atomic_ref with older SYCL code
template <typename T,
          device_address_space address = device_address_space::global>
using device_atomic_ref = sycl::custom_device_atomic_ref<T, address>;

}  // namespace vecmem

#endif  // defined(VECMEM_HAVE_SYCL_ATOMIC_REF)
#elif defined(__CUDACC__)

// Local include(s).
#include "vecmem/memory/details/cuda_device_atomic_ref.hpp"

namespace vecmem {

/// Use @c vecmem::cuda::device_atomic_ref in CUDA device code
template <typename T,
          device_address_space address = device_address_space::global>
using device_atomic_ref = cuda::device_atomic_ref<T, address>;

}  // namespace vecmem

#elif defined(__HIPCC__)

// Local include(s).
#include "vecmem/memory/details/hip_device_atomic_ref.hpp"

namespace vecmem {

/// Use @c vecmem::hip::device_atomic_ref in HIP device code
template <typename T,
          device_address_space address = device_address_space::global>
using device_atomic_ref = hip::device_atomic_ref<T, address>;

}  // namespace vecmem

#elif defined(__cpp_lib_atomic_ref)

namespace vecmem {

/// Use @c std::atomic_ref in host code with C++20
template <typename T, device_address_space = device_address_space::global>
using device_atomic_ref = std::atomic_ref<T>;

}  // namespace vecmem

#elif defined(VECMEM_SUPPORT_POSIX_ATOMIC_REF)

// Local include(s).
#include "vecmem/memory/details/posix_device_atomic_ref.hpp"

namespace vecmem {

/// Use @c vecmem::posix_device_atomic_ref with POSIX threads
template <typename T,
          device_address_space address = device_address_space::global>
using device_atomic_ref = posix_device_atomic_ref<T, address>;

}  // namespace vecmem

#else

// Local include(s).
#include "vecmem/memory/details/dummy_device_atomic_ref.hpp"

namespace vecmem {

/// Use @c vecmem::dummy_device_atomic_ref as a fallback
template <typename T,
          device_address_space address = device_address_space::global>
using device_atomic_ref = dummy_device_atomic_ref<T, address>;

}  // namespace vecmem

#endif

// Test that the selected class would fulfill the atomic_ref concept.
#if __cpp_concepts >= 201907L
#include "vecmem/concepts/atomic_ref.hpp"
static_assert(
    vecmem::concepts::atomic_ref<vecmem::device_atomic_ref<uint32_t> >,
    "Atomic reference on uint32_t is incompletely defined.");
static_assert(
    vecmem::concepts::atomic_ref<vecmem::device_atomic_ref<uint64_t> >,
    "Atomic reference on uint64_t is incompletely defined.");
static_assert(
    vecmem::concepts::atomic_ref<vecmem::device_atomic_ref<std::size_t> >,
    "Atomic reference on std::size_t is incompletely defined.");
#endif
