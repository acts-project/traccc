/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/memory_order.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <type_traits>

namespace vecmem {

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE
dummy_device_atomic_ref<T, address>::dummy_device_atomic_ref(reference ref)
    : m_ptr(&ref) {}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE
dummy_device_atomic_ref<T, address>::dummy_device_atomic_ref(
    const dummy_device_atomic_ref& parent)
    : m_ptr(parent.m_ptr) {}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto dummy_device_atomic_ref<T, address>::operator=(
    value_type data) const -> value_type {

    store(data);
    return data;
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE void dummy_device_atomic_ref<T, address>::store(
    value_type data, memory_order order) const {

    exchange(data, order);
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto dummy_device_atomic_ref<T, address>::load(
    memory_order order) const -> value_type {

    value_type tmp = 0;
    compare_exchange_strong(tmp, tmp, order);
    return tmp;
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto dummy_device_atomic_ref<T, address>::exchange(
    value_type data, memory_order order) const -> value_type {

    value_type tmp = load();
    while (!compare_exchange_strong(tmp, data, order))
        ;
    return tmp;
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE bool
dummy_device_atomic_ref<T, address>::compare_exchange_strong(
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
dummy_device_atomic_ref<T, address>::compare_exchange_strong(
    reference expected, value_type desired, memory_order, memory_order) const {

    // This is **NOT** a sane implementation of CAS!
    value_type old = *m_ptr;
    if (old == expected) {
        *m_ptr = desired;
        return true;
    } else {
        expected = old;
        return false;
    }
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto dummy_device_atomic_ref<T, address>::fetch_add(
    value_type data, memory_order order) const -> value_type {

    value_type tmp = load();
    while (!compare_exchange_strong(tmp, tmp + data, order))
        ;
    return tmp;
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto dummy_device_atomic_ref<T, address>::fetch_sub(
    value_type data, memory_order order) const -> value_type {

    value_type tmp = load();
    while (!compare_exchange_strong(tmp, tmp - data, order))
        ;
    return tmp;
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto dummy_device_atomic_ref<T, address>::fetch_and(
    value_type data, memory_order order) const -> value_type {

    value_type tmp = load();
    while (!compare_exchange_strong(tmp, tmp & data, order))
        ;
    return tmp;
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto dummy_device_atomic_ref<T, address>::fetch_or(
    value_type data, memory_order order) const -> value_type {

    value_type tmp = load();
    while (!compare_exchange_strong(tmp, tmp | data, order))
        ;
    return tmp;
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto dummy_device_atomic_ref<T, address>::fetch_xor(
    value_type data, memory_order order) const -> value_type {

    value_type tmp = load();
    while (!compare_exchange_strong(tmp, tmp ^ data, order))
        ;
    return tmp;
}

}  // namespace vecmem
