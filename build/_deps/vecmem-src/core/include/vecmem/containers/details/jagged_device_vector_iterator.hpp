/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <cstddef>
#include <iterator>
#include <type_traits>

namespace vecmem {
namespace details {

/// Custom iterator for @c vecmem::jagged_device_vector
///
/// In order for @c vecmem::jagged_device_vector to be able to offer
/// iteration over its elements in an efficient and safe way, it needs to use
/// this custom iterator type.
///
/// It takes care of converting between the underlying data type and the
/// type presented towards the users for access to the data. On top of
/// providing all the functionality that an iterator has to.
///
template <typename TYPE>
class jagged_device_vector_iterator {

public:
    /// @name Types describing the underlying data
    /// @{

    /// Type of the data object that we have an array of
    typedef data::vector_view<TYPE> data_type;
    /// Pointer to the data object
    typedef std::add_pointer_t<std::add_const_t<data_type>> data_pointer;

    /// @}

    /// @name Type definitions, mimicking STL iterators
    /// @{

    /// Value type being (virtually) iterated on
    typedef device_vector<TYPE> value_type;
    /// (Pointer) Difference type
    typedef std::ptrdiff_t difference_type;
    /// "Reference" type to the underlying (virtual) value
    typedef value_type reference;

    /// Helper class for returning "pointer-like" objects from the iterator
    ///
    /// Since the iterator returned everything by value as temporary objects,
    /// in order to provide a proper return type for its @c operator->, this
    /// custom type needs to be used.
    ///
    class pointer {

    public:
        /// Constructor from a data pointer
        ///
        /// Used a pointer instead of a reference to make the rest of the code
        /// in @c vecmem::details::jagged_device_vector_iterator as unaware of
        /// the existence of this type as possible.
        ///
        VECMEM_HOST_AND_DEVICE
        explicit pointer(const data_pointer data);

        /// Return a pointer to a device vector (non-const)
        VECMEM_HOST_AND_DEVICE
        value_type* operator->();
        /// Return a pointer to a device vector (const)
        VECMEM_HOST_AND_DEVICE
        const value_type* operator->() const;

    private:
        /// Temporary device vector created on the stack
        value_type m_vec;

    };  // class pointer

    /// @}

    /// Default constructor
    VECMEM_HOST_AND_DEVICE
    jagged_device_vector_iterator();
    /// Constructor from an underlying data object
    VECMEM_HOST_AND_DEVICE
    explicit jagged_device_vector_iterator(data_pointer data);
    /// Constructor from a slightly different underlying data object
    template <typename OTHERTYPE,
              std::enable_if_t<details::is_same_nc<TYPE, OTHERTYPE>::value,
                               bool> = true>
    VECMEM_HOST_AND_DEVICE jagged_device_vector_iterator(
        const data::vector_view<OTHERTYPE>* data);
    /// Copy constructor
    VECMEM_HOST_AND_DEVICE
    jagged_device_vector_iterator(const jagged_device_vector_iterator& parent);
    /// Copy constructor
    template <typename T>
    VECMEM_HOST_AND_DEVICE jagged_device_vector_iterator(
        const jagged_device_vector_iterator<T>& parent);

    /// Copy assignment operator
    VECMEM_HOST_AND_DEVICE
    jagged_device_vector_iterator& operator=(
        const jagged_device_vector_iterator& rhs);

    /// @name Value accessor operators
    /// @{

    /// De-reference the iterator
    VECMEM_HOST_AND_DEVICE
    reference operator*() const;
    /// Use the iterator as a pointer
    VECMEM_HOST_AND_DEVICE
    pointer operator->() const;
    /// Return the value at a specific offset
    VECMEM_HOST_AND_DEVICE
    reference operator[](difference_type n) const;

    /// @}

    /// @name Iterator updating operators
    /// @{

    /// Decrement the underlying iterator (with '++' as a prefix)
    VECMEM_HOST_AND_DEVICE
    jagged_device_vector_iterator& operator++();
    /// Decrement the underlying iterator (wuth '++' as a postfix)
    VECMEM_HOST_AND_DEVICE
    jagged_device_vector_iterator operator++(int);

    /// Increment the underlying iterator (with '--' as a prefix)
    VECMEM_HOST_AND_DEVICE
    jagged_device_vector_iterator& operator--();
    /// Increment the underlying iterator (with '--' as a postfix)
    VECMEM_HOST_AND_DEVICE
    jagged_device_vector_iterator operator--(int);

    /// Decrement the underlying iterator by a specific value
    VECMEM_HOST_AND_DEVICE
    jagged_device_vector_iterator operator+(difference_type n) const;
    /// Decrement the underlying iterator by a specific value
    VECMEM_HOST_AND_DEVICE
    jagged_device_vector_iterator& operator+=(difference_type n);

    /// Increment the underlying iterator by a specific value
    VECMEM_HOST_AND_DEVICE
    jagged_device_vector_iterator operator-(difference_type n) const;
    /// Increment the underlying iterator by a specific value
    VECMEM_HOST_AND_DEVICE
    jagged_device_vector_iterator& operator-=(difference_type n);

    /// @}

    /// @name Comparison operators
    /// @{

    /// Check for the equality of two iterators
    VECMEM_HOST_AND_DEVICE
    bool operator==(const jagged_device_vector_iterator& other) const;
    /// Check for the inequality of two iterators
    VECMEM_HOST_AND_DEVICE
    bool operator!=(const jagged_device_vector_iterator& other) const;

    /// @}

private:
    /// Pointer to the data (in an array)
    data_pointer m_ptr;

};  // class jagged_device_vector_iterator

}  // namespace details
}  // namespace vecmem

namespace std {

/// Specialisation of @c std::iterator_traits
///
/// This is necessary to make @c vecmem::reverse_iterator functional on top
/// of @c vecmem::details::jagged_device_vector_iterator.
///
template <typename T>
struct iterator_traits<vecmem::details::jagged_device_vector_iterator<T>> {
    typedef
        typename vecmem::details::jagged_device_vector_iterator<T>::value_type
            value_type;
    typedef typename vecmem::details::jagged_device_vector_iterator<
        T>::difference_type difference_type;
    typedef typename vecmem::details::jagged_device_vector_iterator<T>::pointer
        pointer;
    typedef
        typename vecmem::details::jagged_device_vector_iterator<T>::reference
            reference;
    typedef std::random_access_iterator_tag iterator_category;
};

}  // namespace std

// Include the implementation.
#include "vecmem/containers/impl/jagged_device_vector_iterator.ipp"
