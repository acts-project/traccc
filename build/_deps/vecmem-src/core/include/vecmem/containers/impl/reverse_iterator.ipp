/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/utils/types.hpp"

namespace vecmem {
namespace details {

template <typename Iterator>
VECMEM_HOST_AND_DEVICE reverse_iterator<Iterator>::reverse_iterator()
    : m_current() {}

template <typename Iterator>
VECMEM_HOST_AND_DEVICE reverse_iterator<Iterator>::reverse_iterator(
    iterator_type itr)
    : m_current(itr) {}

template <typename Iterator>
VECMEM_HOST_AND_DEVICE reverse_iterator<Iterator>::reverse_iterator(
    const reverse_iterator& parent)
    : m_current(parent.m_current) {}

template <typename Iterator>
template <typename T>
VECMEM_HOST_AND_DEVICE reverse_iterator<Iterator>::reverse_iterator(
    const reverse_iterator<T>& parent)
    : m_current(parent.base()) {}

template <typename Iterator>
VECMEM_HOST_AND_DEVICE reverse_iterator<Iterator>&
reverse_iterator<Iterator>::operator=(const reverse_iterator& rhs) {

    // Avoid a self-assignment.
    if (this == &rhs) {
        return *this;
    }

    // Copy the forward iterator from the other object.
    m_current = rhs.m_current;

    // Return this object.
    return *this;
}

template <typename Iterator>
VECMEM_HOST_AND_DEVICE typename reverse_iterator<Iterator>::iterator_type
reverse_iterator<Iterator>::base() const {

    return m_current;
}

template <typename Iterator>
VECMEM_HOST_AND_DEVICE typename reverse_iterator<Iterator>::reference
reverse_iterator<Iterator>::operator*() const {

    iterator_type tmp = m_current;
    return *(--tmp);
}

template <typename Iterator>
VECMEM_HOST_AND_DEVICE typename reverse_iterator<Iterator>::pointer
reverse_iterator<Iterator>::operator->() const {

    iterator_type tmp = m_current;
    return to_pointer(--tmp);
}

template <typename Iterator>
VECMEM_HOST_AND_DEVICE typename reverse_iterator<Iterator>::reference
reverse_iterator<Iterator>::operator[](difference_type n) const {

    return *(*this + n);
}

template <typename Iterator>
VECMEM_HOST_AND_DEVICE reverse_iterator<Iterator>&
reverse_iterator<Iterator>::operator++() {

    --m_current;
    return *this;
}

template <typename Iterator>
VECMEM_HOST_AND_DEVICE reverse_iterator<Iterator>
reverse_iterator<Iterator>::operator++(int) {

    reverse_iterator tmp = *this;
    --m_current;
    return tmp;
}

template <typename Iterator>
VECMEM_HOST_AND_DEVICE reverse_iterator<Iterator>&
reverse_iterator<Iterator>::operator--() {

    ++m_current;
    return *this;
}

template <typename Iterator>
VECMEM_HOST_AND_DEVICE reverse_iterator<Iterator>
reverse_iterator<Iterator>::operator--(int) {

    reverse_iterator tmp = *this;
    ++m_current;
    return tmp;
}

template <typename Iterator>
VECMEM_HOST_AND_DEVICE reverse_iterator<Iterator>
reverse_iterator<Iterator>::operator+(difference_type n) const {

    return reverse_iterator(m_current - n);
}

template <typename Iterator>
VECMEM_HOST_AND_DEVICE reverse_iterator<Iterator>&
reverse_iterator<Iterator>::operator+=(difference_type n) {

    m_current -= n;
    return *this;
}

template <typename Iterator>
VECMEM_HOST_AND_DEVICE reverse_iterator<Iterator>
reverse_iterator<Iterator>::operator-(difference_type n) const {

    return reverse_iterator(m_current + n);
}

template <typename Iterator>
VECMEM_HOST_AND_DEVICE reverse_iterator<Iterator>&
reverse_iterator<Iterator>::operator-=(difference_type n) {

    m_current += n;
    return *this;
}

template <typename Iterator>
template <typename T>
VECMEM_HOST_AND_DEVICE T* reverse_iterator<Iterator>::to_pointer(T* ptr) {

    return ptr;
}

template <typename Iterator>
template <typename T>
VECMEM_HOST_AND_DEVICE typename reverse_iterator<Iterator>::pointer
reverse_iterator<Iterator>::to_pointer(T itr) {

    return itr.operator->();
}

template <typename T>
VECMEM_HOST_AND_DEVICE bool operator==(const reverse_iterator<T>& itr1,
                                       const reverse_iterator<T>& itr2) {

    return (itr1.base() == itr2.base());
}

template <typename T>
VECMEM_HOST_AND_DEVICE bool operator!=(const reverse_iterator<T>& itr1,
                                       const reverse_iterator<T>& itr2) {

    return !(itr1 == itr2);
}

}  // namespace details
}  // namespace vecmem
