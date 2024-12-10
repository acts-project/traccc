/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// HIP include(s).
#if defined(__HIP_DEVICE_COMPILE__)
#include <hip/hip_runtime.h>
#endif

// SYCL include(s).
#if defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp>
#endif

/// Helpers for explicit calls to the SYCL atomic functions
#if defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
#define __VECMEM_SYCL_ATOMIC_CALL0(FNAME, PTR) \
    ::sycl::atomic_##FNAME<value_type>(        \
        ::sycl::atomic<value_type>(::sycl::global_ptr<value_type>(PTR)))
#define __VECMEM_SYCL_ATOMIC_CALL1(FNAME, PTR, ARG1) \
    ::sycl::atomic_##FNAME<value_type>(              \
        ::sycl::atomic<value_type>(::sycl::global_ptr<value_type>(PTR)), ARG1)
#define __VECMEM_SYCL_ATOMIC_CALL2(FNAME, PTR, ARG1, ARG2)                     \
    ::sycl::atomic_##FNAME<value_type>(                                        \
        ::sycl::atomic<value_type>(::sycl::global_ptr<value_type>(PTR)), ARG1, \
        ARG2)
#endif

namespace vecmem {

template <typename T>
VECMEM_HOST_AND_DEVICE atomic<T>::atomic(pointer ptr) : m_ptr(ptr) {}

template <typename T>
VECMEM_HOST_AND_DEVICE void atomic<T>::store(value_type data) {

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!defined(SYCL_LANGUAGE_VERSION))
    volatile pointer addr = m_ptr;
    __threadfence();
    *addr = data;
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    __VECMEM_SYCL_ATOMIC_CALL1(store, m_ptr, data);
#else
    *m_ptr = data;
#endif
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto atomic<T>::load() const -> value_type {

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!defined(SYCL_LANGUAGE_VERSION))
    volatile pointer addr = m_ptr;
    __threadfence();
    const value_type value = *addr;
    __threadfence();
    return value;
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL0(load, m_ptr);
#else
    return *m_ptr;
#endif
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto atomic<T>::exchange(value_type data) -> value_type {

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!defined(SYCL_LANGUAGE_VERSION))
    return atomicExch(m_ptr, data);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL1(exchange, m_ptr, data);
#else
    value_type current_value = *m_ptr;
    *m_ptr = data;
    return current_value;
#endif
}

template <typename T>
VECMEM_HOST_AND_DEVICE bool atomic<T>::compare_exchange_strong(
    reference expected, value_type desired) {

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!defined(SYCL_LANGUAGE_VERSION))
    return atomicCAS(m_ptr, expected, desired);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL2(compare_exchange_strong, m_ptr, expected,
                                      desired);
#else
    if (*m_ptr == expected) {
        *m_ptr = desired;
        return true;
    } else {
        expected = *m_ptr;
        return false;
    }
#endif
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto atomic<T>::fetch_add(value_type data)
    -> value_type {

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!defined(SYCL_LANGUAGE_VERSION))
    return atomicAdd(m_ptr, data);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_add, m_ptr, data);
#else
    const value_type result = *m_ptr;
    *m_ptr += data;
    return result;
#endif
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto atomic<T>::fetch_sub(value_type data)
    -> value_type {

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!defined(SYCL_LANGUAGE_VERSION))
    return atomicSub(m_ptr, data);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_sub, m_ptr, data);
#else
    const value_type result = *m_ptr;
    *m_ptr -= data;
    return result;
#endif
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto atomic<T>::fetch_and(value_type data)
    -> value_type {

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!defined(SYCL_LANGUAGE_VERSION))
    return atomicAnd(m_ptr, data);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_and, m_ptr, data);
#else
    const value_type result = *m_ptr;
    *m_ptr &= data;
    return result;
#endif
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto atomic<T>::fetch_or(value_type data) -> value_type {

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!defined(SYCL_LANGUAGE_VERSION))
    return atomicOr(m_ptr, data);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_or, m_ptr, data);
#else
    const value_type result = *m_ptr;
    *m_ptr |= data;
    return result;
#endif
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto atomic<T>::fetch_xor(value_type data)
    -> value_type {

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!defined(SYCL_LANGUAGE_VERSION))
    return atomicXor(m_ptr, data);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_xor, m_ptr, data);
#else
    const value_type result = *m_ptr;
    *m_ptr ^= data;
    return result;
#endif
}

}  // namespace vecmem
