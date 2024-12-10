/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// VecMem include(s).
#include "vecmem/utils/types.hpp"

// System include(s).
#include <type_traits>

// Provide a different implementation for modern SYCL and everything else
#if (defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)) && \
    defined(VECMEM_HAVE_SYCL_ATOMIC_REF)

namespace vecmem {

/// In modern SYCL code @c vecmem::atomic is an alias for @c sycl::atomic_ref
///
/// It has to be an actual class and not just a typedef, because
/// @c sycl::atomic_ref, as the name implies, is created on top of references,
/// and not on top of pointers.
///
template <typename T>
class atomic
    : public ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed,
                                ::sycl::memory_scope::device,
                                ::sycl::access::address_space::global_space> {

public:
    /// @name Type definitions
    /// @{

    /// Type managed by the object
    typedef T value_type;
    /// Difference between two objects
    typedef value_type difference_type;
    /// Pointer to the value in global memory
    typedef value_type* pointer;
    /// Reference to a value given by the user
    typedef value_type& reference;

    /// @}

    /// Constructor, with a pointer to the managed variable
    VECMEM_HOST_AND_DEVICE
    atomic(pointer ptr)
        : ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed,
                             ::sycl::memory_scope::device,
                             ::sycl::access::address_space::global_space>(
              *ptr) {}

};  // class atomic

}  // namespace vecmem

#else

namespace vecmem {

/// Class providing atomic operations for the VecMem code
///
/// It is only meant to be used with primitive types. Ones that CUDA, HIP and
/// SYCL built-in functions exist for. So no structs, or even pointers.
///
/// Note that it does not perform atomic operations in host code! That may
/// be implemented with @c std::atomic_ref in C++20 when VecMem switches to
/// that standard. But for now all operations in host code are performed as
/// "regular" operations.
///
template <typename T>
class atomic {

public:
    /// @name Type definitions
    /// @{

    /// Type managed by the object
    typedef T value_type;
    /// Difference between two objects
    typedef value_type difference_type;
    /// Pointer to the value in global memory
    typedef value_type* pointer;
    /// Reference to a value given by the user
    typedef value_type& reference;

    /// @}

    /// @name Check(s) on the value type
    /// @{

    static_assert(std::is_integral<value_type>::value,
                  "vecmem::atomic only accepts built-in integral types");

    /// @}

    /// Constructor, with a pointer to the managed variable
    VECMEM_HOST_AND_DEVICE
    atomic(pointer ptr);

    /// @name Value setter/getter functions
    /// @{

    /// Set the variable to the desired value
    VECMEM_HOST_AND_DEVICE
    void store(value_type data);
    /// Get the value of the variable
    VECMEM_HOST_AND_DEVICE
    value_type load() const;

    /// Exchange the current value of the variable with a different one
    VECMEM_HOST_AND_DEVICE
    value_type exchange(value_type data);

    /// Compare against the current value, and exchange only if different
    VECMEM_HOST_AND_DEVICE
    bool compare_exchange_strong(reference expected, value_type desired);

    /// @}

    /// @name Value modifier functions
    /// @{

    /// Add a chosen value to the stored variable
    VECMEM_HOST_AND_DEVICE
    value_type fetch_add(value_type data);
    /// Substitute a chosen value from the stored variable
    VECMEM_HOST_AND_DEVICE
    value_type fetch_sub(value_type data);

    /// Replace the current value with the specified value AND-ed to it
    VECMEM_HOST_AND_DEVICE
    value_type fetch_and(value_type data);
    /// Replace the current value with the specified value OR-d to it
    VECMEM_HOST_AND_DEVICE
    value_type fetch_or(value_type data);
    /// Replace the current value with the specified value XOR-d to it
    VECMEM_HOST_AND_DEVICE
    value_type fetch_xor(value_type data);

    /// @}

private:
    /// Pointer to the value to perform atomic operations on
    pointer m_ptr;

};  // class atomic

}  // namespace vecmem

// Include the implementation.
#include "vecmem/memory/impl/atomic.ipp"

#endif  // sycl::atomic_ref
