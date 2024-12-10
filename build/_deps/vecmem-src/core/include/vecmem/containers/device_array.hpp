/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/containers/details/reverse_iterator.hpp"
#include "vecmem/utils/type_traits.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <cstddef>
#include <type_traits>

namespace vecmem {

/// Class mimicking a host-filled @c std::array in "device code"
///
/// Unlike @c vecmem::array, its size must be defined at compile time. Since
/// for arrays whose size is defined at runtime, one can use
/// @c vecmem::device_vector instead.
///
template <typename T, std::size_t N>
class device_array {

public:
    /// @name Type definitions, mimicking @c std::array
    /// @{

    /// Type of the array elements
    typedef T value_type;
    /// Size type for the array
    typedef std::size_t size_type;
    /// Pointer difference type
    typedef std::ptrdiff_t difference_type;

    /// Value reference type
    typedef value_type& reference;
    /// Constant value reference type
    typedef const value_type& const_reference;
    /// Value pointer type
    typedef value_type* pointer;
    /// Constant value pointer type
    typedef const value_type* const_pointer;

    /// Forward iterator type
    typedef pointer iterator;
    /// Constant forward iterator type
    typedef const_pointer const_iterator;
    /// Reverse iterator type
    typedef vecmem::details::reverse_iterator<iterator> reverse_iterator;
    /// Constant reverse iterator type
    typedef vecmem::details::reverse_iterator<const_iterator>
        const_reverse_iterator;

    /// @}

    /// Constructor, on top of a previously allocated/filled block of memory
    VECMEM_HOST_AND_DEVICE
    explicit device_array(const data::vector_view<value_type>& data);
    /// Construct a const device array from a non-const data object
    template <
        typename OTHERTYPE,
        std::enable_if_t<details::is_same_nc<T, OTHERTYPE>::value, bool> = true>
    VECMEM_HOST_AND_DEVICE explicit device_array(
        const data::vector_view<OTHERTYPE>& data);
    /// Copy constructor
    VECMEM_HOST_AND_DEVICE
    device_array(const device_array& parent);

    /// Copy assignment operator
    VECMEM_HOST_AND_DEVICE
    device_array& operator=(const device_array& rhs);

    /// @name Array element access functions
    /// @{

    /// Return a specific element of the array in a "safe way" (non-const)
    VECMEM_HOST_AND_DEVICE
    reference at(size_type pos);
    /// Return a specific element of the array in a "safe way" (const)
    VECMEM_HOST_AND_DEVICE
    const_reference at(size_type pos) const;

    /// Return a specific element of the array (non-const)
    VECMEM_HOST_AND_DEVICE
    reference operator[](size_type pos);
    /// Return a specific element of the array (const)
    VECMEM_HOST_AND_DEVICE
    const_reference operator[](size_type pos) const;

    /// Return the first element of the array (non-const)
    VECMEM_HOST_AND_DEVICE
    reference front();
    /// Return the first element of the array (const)
    VECMEM_HOST_AND_DEVICE
    const_reference front() const;

    /// Return the last element of the array (non-const)
    VECMEM_HOST_AND_DEVICE
    reference back();
    /// Return the last element of the array (const)
    VECMEM_HOST_AND_DEVICE
    const_reference back() const;

    /// Access the underlying memory array (non-const)
    VECMEM_HOST_AND_DEVICE
    pointer data();
    /// Access the underlying memory array (const)
    VECMEM_HOST_AND_DEVICE
    const_pointer data() const;

    /// @}

    /// @name Iterator providing functions
    /// @{

    /// Return a forward iterator pointing at the beginning of the array
    VECMEM_HOST_AND_DEVICE
    iterator begin();
    /// Return a constant forward iterator pointing at the beginning of the
    /// array
    VECMEM_HOST_AND_DEVICE
    const_iterator begin() const;
    /// Return a constant forward iterator pointing at the beginning of the
    /// array
    VECMEM_HOST_AND_DEVICE
    const_iterator cbegin() const;

    /// Return a forward iterator pointing at the end of the array
    VECMEM_HOST_AND_DEVICE
    iterator end();
    /// Return a constant forward iterator pointing at the end of the array
    VECMEM_HOST_AND_DEVICE
    const_iterator end() const;
    /// Return a constant forward iterator pointing at the end of the array
    VECMEM_HOST_AND_DEVICE
    const_iterator cend() const;

    /// Return a reverse iterator pointing at the end of the array
    VECMEM_HOST_AND_DEVICE
    reverse_iterator rbegin();
    /// Return a constant reverse iterator pointing at the end of the array
    VECMEM_HOST_AND_DEVICE
    const_reverse_iterator rbegin() const;
    /// Return a constant reverse iterator pointing at the end of the array
    VECMEM_HOST_AND_DEVICE
    const_reverse_iterator crbegin() const;

    /// Return a reverse iterator pointing at the beginning of the array
    VECMEM_HOST_AND_DEVICE
    reverse_iterator rend();
    /// Return a constant reverse iterator pointing at the beginning of the
    /// array
    VECMEM_HOST_AND_DEVICE
    const_reverse_iterator rend() const;
    /// Return a constant reverse iterator pointing at the beginning of the
    /// array
    VECMEM_HOST_AND_DEVICE
    const_reverse_iterator crend() const;

    /// @}

    /// @name Capacity checking functions
    /// @{

    /// Check whether the array is empty
    VECMEM_HOST_AND_DEVICE
    constexpr bool empty() const;
    /// Return the number of elements in the array
    VECMEM_HOST_AND_DEVICE
    constexpr size_type size() const;
    /// Return the maximum (fixed) number of elements in the array
    VECMEM_HOST_AND_DEVICE
    constexpr size_type max_size() const;

    /// @}

private:
    /// Pointer to the start of the array
    pointer m_ptr;

};  // class device_array

}  // namespace vecmem

// Include the implementation.
#include "vecmem/containers/impl/device_array.ipp"
