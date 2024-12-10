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
namespace details {

/// @brief Convert a memory order to the corresponding builtin memory order
/// @param o The (vecmem) memory order
/// @return The corresponding builtin memory order
///
constexpr int memorder_to_posix_builtin(memory_order o) {
    switch (o) {
        case memory_order::relaxed:
            return __ATOMIC_RELAXED;
        case memory_order::consume:
            return __ATOMIC_CONSUME;
        case memory_order::acquire:
            return __ATOMIC_ACQUIRE;
        case memory_order::release:
            return __ATOMIC_RELEASE;
        case memory_order::acq_rel:
            return __ATOMIC_ACQ_REL;
        case memory_order::seq_cst:
            return __ATOMIC_SEQ_CST;
        default:
            assert(false);
            return 0;
    }
}

}  // namespace details

template <typename T, device_address_space address>
VECMEM_HOST posix_device_atomic_ref<T, address>::posix_device_atomic_ref(
    reference ref)
    : m_ptr(&ref) {}

template <typename T, device_address_space address>
VECMEM_HOST posix_device_atomic_ref<T, address>::posix_device_atomic_ref(
    const posix_device_atomic_ref& parent)
    : m_ptr(parent.m_ptr) {}

template <typename T, device_address_space address>
VECMEM_HOST auto posix_device_atomic_ref<T, address>::operator=(
    value_type data) const -> value_type {

    store(data);
    return data;
}

template <typename T, device_address_space address>
VECMEM_HOST void posix_device_atomic_ref<T, address>::store(
    value_type data, memory_order order) const {

    __atomic_store_n(m_ptr, data, details::memorder_to_posix_builtin(order));
}

template <typename T, device_address_space address>
VECMEM_HOST auto posix_device_atomic_ref<T, address>::load(
    memory_order order) const -> value_type {

    return __atomic_load_n(m_ptr, details::memorder_to_posix_builtin(order));
}

template <typename T, device_address_space address>
VECMEM_HOST auto posix_device_atomic_ref<T, address>::exchange(
    value_type data, memory_order order) const -> value_type {

    return __atomic_exchange_n(m_ptr, data,
                               details::memorder_to_posix_builtin(order));
}

template <typename T, device_address_space address>
VECMEM_HOST bool posix_device_atomic_ref<T, address>::compare_exchange_strong(
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
VECMEM_HOST bool posix_device_atomic_ref<T, address>::compare_exchange_strong(
    reference expected, value_type desired, memory_order success,
    memory_order failure) const {

    assert(failure != memory_order::release &&
           failure != memory_order::acq_rel);

    return __atomic_compare_exchange_n(
        m_ptr, &expected, desired, false,
        details::memorder_to_posix_builtin(success),
        details::memorder_to_posix_builtin(failure));
}

template <typename T, device_address_space address>
VECMEM_HOST auto posix_device_atomic_ref<T, address>::fetch_add(
    value_type data, memory_order order) const -> value_type {

    return __atomic_fetch_add(m_ptr, data,
                              details::memorder_to_posix_builtin(order));
}

template <typename T, device_address_space address>
VECMEM_HOST auto posix_device_atomic_ref<T, address>::fetch_sub(
    value_type data, memory_order order) const -> value_type {

    return __atomic_fetch_add(m_ptr, -data,
                              details::memorder_to_posix_builtin(order));
}

template <typename T, device_address_space address>
VECMEM_HOST auto posix_device_atomic_ref<T, address>::fetch_and(
    value_type data, memory_order order) const -> value_type {

    return __atomic_fetch_and(m_ptr, data,
                              details::memorder_to_posix_builtin(order));
}

template <typename T, device_address_space address>
VECMEM_HOST auto posix_device_atomic_ref<T, address>::fetch_or(
    value_type data, memory_order order) const -> value_type {

    return __atomic_fetch_or(m_ptr, data,
                             details::memorder_to_posix_builtin(order));
}

template <typename T, device_address_space address>
VECMEM_HOST auto posix_device_atomic_ref<T, address>::fetch_xor(
    value_type data, memory_order order) const -> value_type {

    return __atomic_fetch_xor(m_ptr, data,
                              details::memorder_to_posix_builtin(order));
}

}  // namespace vecmem
