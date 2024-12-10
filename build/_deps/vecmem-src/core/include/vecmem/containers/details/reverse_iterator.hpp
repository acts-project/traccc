/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/utils/types.hpp"

// System include(s).
#include <iterator>

namespace vecmem {
namespace details {

/// Type mimicking @c std::reverse_iterator
///
/// It behaves and is implemented basically in the same way as
/// @c std::reverse_iterator. The only reason why it needs to exist is to
/// make it possible to be used from device code. (The STL type can not be.)
///
template <typename Iterator>
class reverse_iterator {

public:
    /// @name Type definitions, mimicking @c std::reverse_iterator
    /// @{

    /// Type of the forward iterator, being wrapped
    typedef Iterator iterator_type;
    /// Value type being iterated on
    typedef typename std::iterator_traits<iterator_type>::value_type value_type;
    /// (Pointer) Difference type
    typedef typename std::iterator_traits<iterator_type>::difference_type
        difference_type;
    /// Pointer type to the underlying value
    typedef typename std::iterator_traits<iterator_type>::pointer pointer;
    /// Reference type to the underlying value
    typedef typename std::iterator_traits<iterator_type>::reference reference;

    /// @}

    /// Default constructor
    VECMEM_HOST_AND_DEVICE
    reverse_iterator();
    /// Constructor, from a forward iterator
    VECMEM_HOST_AND_DEVICE
    reverse_iterator(iterator_type itr);

    /// Copy constructor
    VECMEM_HOST_AND_DEVICE
    reverse_iterator(const reverse_iterator& parent);
    /// Copy constructor
    template <typename T>
    VECMEM_HOST_AND_DEVICE reverse_iterator(const reverse_iterator<T>& parent);

    /// Copy assignment operator
    VECMEM_HOST_AND_DEVICE
    reverse_iterator& operator=(const reverse_iterator& rhs);

    /// Access the base/forward iterator
    VECMEM_HOST_AND_DEVICE
    iterator_type base() const;

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
    reverse_iterator& operator++();
    /// Decrement the underlying iterator (wuth '++' as a postfix)
    VECMEM_HOST_AND_DEVICE
    reverse_iterator operator++(int);

    /// Increment the underlying iterator (with '--' as a prefix)
    VECMEM_HOST_AND_DEVICE
    reverse_iterator& operator--();
    /// Increment the underlying iterator (with '--' as a postfix)
    VECMEM_HOST_AND_DEVICE
    reverse_iterator operator--(int);

    /// Decrement the underlying iterator by a specific value
    VECMEM_HOST_AND_DEVICE
    reverse_iterator operator+(difference_type n) const;
    /// Decrement the underlying iterator by a specific value
    VECMEM_HOST_AND_DEVICE
    reverse_iterator& operator+=(difference_type n);

    /// Increment the underlying iterator by a specific value
    VECMEM_HOST_AND_DEVICE
    reverse_iterator operator-(difference_type n) const;
    /// Increment the underlying iterator by a specific value
    VECMEM_HOST_AND_DEVICE
    reverse_iterator& operator-=(difference_type n);

    /// @}

private:
    /// Helper function producing a pointer from a forward iterator
    template <typename T>
    VECMEM_HOST_AND_DEVICE static T* to_pointer(T* ptr);
    /// Helper function producing a pointer from a forward iterator
    template <typename T>
    VECMEM_HOST_AND_DEVICE static pointer to_pointer(T itr);

    /// Forward iterator type to the "current" element
    iterator_type m_current;

};  // class reverse_iterator

/// Comparison operator for reverse iterators
template <typename T>
VECMEM_HOST_AND_DEVICE bool operator==(const reverse_iterator<T>& itr1,
                                       const reverse_iterator<T>& itr2);
/// Comparison operator for reverse iterators
template <typename T>
VECMEM_HOST_AND_DEVICE bool operator!=(const reverse_iterator<T>& itr1,
                                       const reverse_iterator<T>& itr2);

}  // namespace details
}  // namespace vecmem

// Include the implementation.
#include "vecmem/containers/impl/reverse_iterator.ipp"
