/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <cassert>

namespace vecmem {
namespace cuda {

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE device_atomic_ref<T, address>::device_atomic_ref(
    reference ref)
    : m_ptr(&ref) {}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE device_atomic_ref<T, address>::device_atomic_ref(
    const device_atomic_ref& parent)
    : m_ptr(parent.m_ptr) {}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::operator=(
    value_type data) const -> value_type {

    store(data);
    return data;
}

// Only invoke __threadfence() during device code compilation. Without this,
// nvcc gets upset about calling this **device only** function from a function
// labeled HOST_AND_DEVICE. Allow an outside source to set the macro, so that
// vecmem::hip::device_atomic_ref could have its own logic for setting it up
// correctly.
#ifndef __VECMEM_THREADFENCE
#ifdef __CUDA_ARCH__
#define __VECMEM_THREADFENCE __threadfence()
#else
#define __VECMEM_THREADFENCE
#endif  // defined(__CUDA_ARCH__)
#endif  // not defined(__VECMEM_THREADFENCE)

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE void device_atomic_ref<T, address>::store(
    value_type data, memory_order) const {

    volatile pointer addr = m_ptr;
    __VECMEM_THREADFENCE;
    *addr = data;
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::load(
    memory_order) const -> value_type {

    volatile pointer addr = m_ptr;
    __VECMEM_THREADFENCE;
    const value_type value = *addr;
    __VECMEM_THREADFENCE;
    return value;
}

#undef __VECMEM_THREADFENCE

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::exchange(
    value_type data, memory_order) const -> value_type {

    return atomicExch(m_ptr, data);
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE bool
device_atomic_ref<T, address>::compare_exchange_strong(
    reference expected, value_type desired, memory_order order) const {

    if (order == memory_order::acq_rel) {
        return compare_exchange_strong(expected, desired, order,
                                       memory_order::acquire);
    } else if (order == memory_order::release) {
        return compare_exchange_strong(expected, desired, order,
                                       memory_order::relaxed);
    } else {
        return compare_exchange_strong(expected, desired, order, order);
    }
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE bool
device_atomic_ref<T, address>::compare_exchange_strong(
    reference expected, value_type desired, memory_order,
    memory_order failure) const {

    (void)failure;
    assert(failure != memory_order::release &&
           failure != memory_order::acq_rel);

    const value_type r = atomicCAS(m_ptr, expected, desired);
    // atomicCAS returns the old value, so the change will have succeeded if
    // the old value was the expected value.
    if (r == expected) {
        return true;
    } else {
        expected = r;
        return false;
    }
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_add(
    value_type data, memory_order) const -> value_type {

    return atomicAdd(m_ptr, data);
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_sub(
    value_type data, memory_order) const -> value_type {

    return atomicSub(m_ptr, data);
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_and(
    value_type data, memory_order) const -> value_type {

    return atomicAnd(m_ptr, data);
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_or(
    value_type data, memory_order) const -> value_type {

    return atomicOr(m_ptr, data);
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_xor(
    value_type data, memory_order) const -> value_type {

    return atomicXor(m_ptr, data);
}

}  // namespace cuda
}  // namespace vecmem
