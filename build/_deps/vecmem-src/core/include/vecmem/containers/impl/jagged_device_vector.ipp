/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <cassert>

namespace vecmem {

template <typename T>
VECMEM_HOST_AND_DEVICE jagged_device_vector<T>::jagged_device_vector(
    data::jagged_vector_view<T> data)
    : m_size(data.size()), m_ptr(data.ptr()) {}

template <typename T>
VECMEM_HOST_AND_DEVICE jagged_device_vector<T>::jagged_device_vector(
    const jagged_device_vector& parent)
    : m_size(parent.m_size), m_ptr(parent.m_ptr) {}

template <typename T>
VECMEM_HOST_AND_DEVICE jagged_device_vector<T>&
jagged_device_vector<T>::operator=(const jagged_device_vector& rhs) {

    // Check if anything needs to be done.
    if (this == &rhs) {
        return *this;
    }

    // Make this object point at the same data in memory as the one we're
    // copying from.
    m_size = rhs.m_size;
    m_ptr = rhs.m_ptr;

    // Return this object.
    return *this;
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::at(size_type pos)
    -> reference {

    // Check if the index is valid.
    assert(pos < m_size);

    // Return a reference to the vector element.
    return reference{m_ptr[pos]};
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::at(size_type pos) const
    -> const_reference {

    // Check if the index is valid.
    assert(pos < m_size);

    // Return a reference to the vector element.
    return const_reference{m_ptr[pos]};
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::operator[](size_type pos)
    -> reference {

    // Return a reference to the vector element.
    return reference{m_ptr[pos]};
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::operator[](
    size_type pos) const -> const_reference {

    // Return a reference to the vector element.
    return const_reference{m_ptr[pos]};
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::front() -> reference {

    // Make sure that there is at least one element in the outer vector.
    assert(m_size > 0);

    // Return a reference to the first element of the vector.
    return reference{m_ptr[0]};
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::front() const
    -> const_reference {

    // Make sure that there is at least one element in the outer vector.
    assert(m_size > 0);

    // Return a reference to the first element of the vector.
    return const_reference{m_ptr[0]};
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::back() -> reference {

    // Make sure that there is at least one element in the outer vector.
    assert(m_size > 0);

    // Return a reference to the last element of the vector.
    return reference{m_ptr[m_size - 1]};
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::back() const
    -> const_reference {

    // Make sure that there is at least one element in the outer vector.
    assert(m_size > 0);

    // Return a reference to the last element of the vector.
    return const_reference{m_ptr[m_size - 1]};
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::begin() -> iterator {

    return iterator(m_ptr);
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::begin() const
    -> const_iterator {

    return const_iterator(m_ptr);
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::cbegin() const
    -> const_iterator {

    return begin();
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::end() -> iterator {

    return iterator(m_ptr + m_size);
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::end() const
    -> const_iterator {

    return const_iterator(m_ptr + m_size);
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::cend() const
    -> const_iterator {

    return end();
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::rbegin()
    -> reverse_iterator {

    return reverse_iterator(end());
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::rbegin() const
    -> const_reverse_iterator {

    return const_reverse_iterator(end());
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::crbegin() const
    -> const_reverse_iterator {

    return rbegin();
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::rend()
    -> reverse_iterator {

    return reverse_iterator(begin());
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::rend() const
    -> const_reverse_iterator {

    return const_reverse_iterator(begin());
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::crend() const
    -> const_reverse_iterator {

    return rend();
}

template <typename T>
VECMEM_HOST_AND_DEVICE bool jagged_device_vector<T>::empty(void) const {
    return m_size == 0;
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::size(void) const
    -> size_type {
    return m_size;
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::max_size() const
    -> size_type {

    return m_size;
}

template <typename T>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector<T>::capacity() const
    -> size_type {

    return m_size;
}

}  // namespace vecmem
