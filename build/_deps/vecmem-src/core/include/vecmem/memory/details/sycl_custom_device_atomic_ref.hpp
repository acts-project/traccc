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
#include "vecmem/memory/memory_order.hpp"

// System include(s).
#include <type_traits>

namespace vecmem {
namespace sycl {

/// Custom implementation for atomic operations in SYCL device code
///
/// @tparam T Type to perform atomic operations on
/// @tparam address The device address space to use
///
template <typename T,
          device_address_space address = device_address_space::global>
class custom_device_atomic_ref {

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

    static_assert(
        std::is_integral<value_type>::value,
        "vecmem::sycl::custom_device_atomic_ref only accepts built-in "
        "integral types");

    /// @}

    /// Constructor, with a pointer to the managed variable
    explicit custom_device_atomic_ref(reference ref);
    /// Copy constructor
    custom_device_atomic_ref(const custom_device_atomic_ref& parent);

    /// Disable the assignment operator
    custom_device_atomic_ref& operator=(const custom_device_atomic_ref&) =
        delete;

    /// @name Value setter/getter functions
    /// @{

    /// Assigns a value desired to the referenced object
    ///
    /// @see vecmem::device_atomic_ref::store
    ///
    value_type operator=(value_type data) const;

    /// Set the variable to the desired value
    void store(value_type data,
               memory_order order = memory_order::seq_cst) const;
    /// Get the value of the variable
    value_type load(memory_order order = memory_order::seq_cst) const;

    /// Exchange the current value of the variable with a different one
    value_type exchange(value_type data,
                        memory_order order = memory_order::seq_cst) const;

    /// Compare against the current value, and exchange only if different
    bool compare_exchange_strong(reference expected, value_type desired,
                                 memory_order success,
                                 memory_order failure) const;
    /// Compare against the current value, and exchange only if different
    bool compare_exchange_strong(
        reference expected, value_type desired,
        memory_order order = memory_order::seq_cst) const;

    /// @}

    /// @name Value modifier functions
    /// @{

    /// Add a chosen value to the stored variable
    value_type fetch_add(value_type data,
                         memory_order order = memory_order::seq_cst) const;
    /// Substitute a chosen value from the stored variable
    value_type fetch_sub(value_type data,
                         memory_order order = memory_order::seq_cst) const;

    /// Replace the current value with the specified value AND-ed to it
    value_type fetch_and(value_type data,
                         memory_order order = memory_order::seq_cst) const;
    /// Replace the current value with the specified value OR-d to it
    value_type fetch_or(value_type data,
                        memory_order order = memory_order::seq_cst) const;
    /// Replace the current value with the specified value XOR-d to it
    value_type fetch_xor(value_type data,
                         memory_order order = memory_order::seq_cst) const;

    /// @}

private:
    /// Pointer to the value to perform atomic operations on
    pointer m_ptr;

};  // class custom_device_atomic_ref

}  // namespace sycl
}  // namespace vecmem

// Include the implementation.
#include "vecmem/memory/impl/sycl_custom_device_atomic_ref.ipp"
