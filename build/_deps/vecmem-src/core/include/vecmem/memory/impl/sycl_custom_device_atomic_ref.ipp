/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// SYCL include(s).
#include <sycl/sycl.hpp>

namespace vecmem {
namespace sycl {
namespace details {

template <device_address_space address>
struct custom_address_space {};

template <>
struct custom_address_space<device_address_space::global> {
    static constexpr ::sycl::access::address_space add =
        ::sycl::access::address_space::global_space;

    template <typename T>
    using ptr_t = ::sycl::global_ptr<T>;
};

template <>
struct custom_address_space<device_address_space::local> {
    static constexpr ::sycl::access::address_space add =
        ::sycl::access::address_space::local_space;
    template <typename T>
    using ptr_t = ::sycl::local_ptr<T>;
};

}  // namespace details

#define __VECMEM_SYCL_ATOMIC_CALL0(FNAME, PTR)                               \
    ::sycl::atomic_##FNAME<value_type,                                       \
                           details::custom_address_space<address>::add>(     \
        ::sycl::atomic<value_type,                                           \
                       details::custom_address_space<address>::add>(         \
            typename details::custom_address_space<address>::template ptr_t< \
                value_type>(PTR)))
#define __VECMEM_SYCL_ATOMIC_CALL1(FNAME, PTR, ARG1)                         \
    ::sycl::atomic_##FNAME<value_type,                                       \
                           details::custom_address_space<address>::add>(     \
        ::sycl::atomic<value_type,                                           \
                       details::custom_address_space<address>::add>(         \
            typename details::custom_address_space<address>::template ptr_t< \
                value_type>(PTR)),                                           \
        ARG1)
#define __VECMEM_SYCL_ATOMIC_CALL2(FNAME, PTR, ARG1, ARG2)                   \
    ::sycl::atomic_##FNAME<value_type,                                       \
                           details::custom_address_space<address>::add>(     \
        ::sycl::atomic<value_type,                                           \
                       details::custom_address_space<address>::add>(         \
            typename details::custom_address_space<address>::template ptr_t< \
                value_type>(PTR)),                                           \
        ARG1, ARG2)

template <typename T, device_address_space address>
custom_device_atomic_ref<T, address>::custom_device_atomic_ref(reference ref)
    : m_ptr(&ref) {}

template <typename T, device_address_space address>
custom_device_atomic_ref<T, address>::custom_device_atomic_ref(
    const custom_device_atomic_ref& parent)
    : m_ptr(parent.m_ptr) {}

template <typename T, device_address_space address>
auto custom_device_atomic_ref<T, address>::operator=(value_type data) const
    -> value_type {

    store(data);
    return data;
}

template <typename T, device_address_space address>
void custom_device_atomic_ref<T, address>::store(value_type data,
                                                 memory_order) const {

    __VECMEM_SYCL_ATOMIC_CALL1(store, m_ptr, data);
}

template <typename T, device_address_space address>
auto custom_device_atomic_ref<T, address>::load(memory_order) const
    -> value_type {

    return __VECMEM_SYCL_ATOMIC_CALL0(load, m_ptr);
}

template <typename T, device_address_space address>
auto custom_device_atomic_ref<T, address>::exchange(value_type data,
                                                    memory_order) const
    -> value_type {

    return __VECMEM_SYCL_ATOMIC_CALL1(exchange, m_ptr, data);
}

template <typename T, device_address_space address>
bool custom_device_atomic_ref<T, address>::compare_exchange_strong(
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
bool custom_device_atomic_ref<T, address>::compare_exchange_strong(
    reference expected, value_type desired, memory_order,
    memory_order failure) const {

    (void)failure;
    assert(failure != memory_order::release &&
           failure != memory_order::acq_rel);

    return __VECMEM_SYCL_ATOMIC_CALL2(compare_exchange_strong, m_ptr, expected,
                                      desired);
}

template <typename T, device_address_space address>
auto custom_device_atomic_ref<T, address>::fetch_add(value_type data,
                                                     memory_order) const
    -> value_type {

    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_add, m_ptr, data);
}

template <typename T, device_address_space address>
auto custom_device_atomic_ref<T, address>::fetch_sub(value_type data,
                                                     memory_order order) const
    -> value_type {

    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_sub, m_ptr, data);
}

template <typename T, device_address_space address>
auto custom_device_atomic_ref<T, address>::fetch_and(value_type data,
                                                     memory_order) const
    -> value_type {

    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_and, m_ptr, data);
}

template <typename T, device_address_space address>
auto custom_device_atomic_ref<T, address>::fetch_or(value_type data,
                                                    memory_order) const
    -> value_type {

    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_or, m_ptr, data);
}

template <typename T, device_address_space address>
auto custom_device_atomic_ref<T, address>::fetch_xor(value_type data,
                                                     memory_order) const
    -> value_type {

    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_xor, m_ptr, data);
}

#undef __VECMEM_SYCL_ATOMIC_CALL0
#undef __VECMEM_SYCL_ATOMIC_CALL1
#undef __VECMEM_SYCL_ATOMIC_CALL2

}  // namespace sycl
}  // namespace vecmem
