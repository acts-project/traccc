/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// HIP include(s)
#if defined(__HIP_DEVICE_COMPILE__)
#include <hip/hip_runtime.h>
#endif

// System include(s).
#include <cassert>

namespace vecmem {

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE device_array<T, N>::device_array(
    const data::vector_view<value_type>& data)
    : m_ptr(data.ptr()) {

    assert(data.size() >= N);
}

template <typename T, std::size_t N>
template <typename OTHERTYPE,
          std::enable_if_t<details::is_same_nc<T, OTHERTYPE>::value, bool> >
VECMEM_HOST_AND_DEVICE device_array<T, N>::device_array(
    const data::vector_view<OTHERTYPE>& data)
    : m_ptr(data.ptr()) {

    assert(data.size() >= N);
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE device_array<T, N>::device_array(
    const device_array& parent)
    : m_ptr(parent.m_ptr) {}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE device_array<T, N>& device_array<T, N>::operator=(
    const device_array& rhs) {

    // Prevent self-assignment.
    if (this == &rhs) {
        return *this;
    }

    // Copy the other object's payload.
    m_ptr = rhs.m_ptr;

    // Return a reference to this object.
    return *this;
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::at(size_type pos) -> reference {

    // Check if the index is valid.
    assert(pos < N);

    // Return a reference to the vector element.
    return m_ptr[pos];
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::at(size_type pos) const
    -> const_reference {

    // Check if the index is valid.
    assert(pos < N);

    // Return a reference to the vector element.
    return m_ptr[pos];
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::operator[](size_type pos)
    -> reference {

    // Return a reference to the vector element.
    return m_ptr[pos];
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::operator[](size_type pos) const
    -> const_reference {

    // Return a reference to the vector element.
    return m_ptr[pos];
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::front() -> reference {

    // Make sure that there is at least one element in the vector.
    static_assert(N > 0, "Cannot return first element of empty array");

    // Return a reference to the first element of the vector.
    return m_ptr[0];
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::front() const
    -> const_reference {

    // Make sure that there is at least one element in the vector.
    static_assert(N > 0, "Cannot return first element of empty array");

    // Return a reference to the first element of the vector.
    return m_ptr[0];
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::back() -> reference {

    // Make sure that there is at least one element in the vector.
    static_assert(N > 0, "Cannot return last element of empty array");

    // Return a reference to the last element of the vector.
    return m_ptr[N - 1];
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::back() const
    -> const_reference {

    // Make sure that there is at least one element in the vector.
    static_assert(N > 0, "Cannot return last element of empty array");

    // Return a reference to the last element of the vector.
    return m_ptr[N - 1];
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::data() -> pointer {

    return m_ptr;
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::data() const -> const_pointer {

    return m_ptr;
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::begin() -> iterator {

    return iterator(m_ptr);
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::begin() const
    -> const_iterator {

    return const_iterator(m_ptr);
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::cbegin() const
    -> const_iterator {

    return begin();
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::end() -> iterator {

    return iterator(m_ptr + N);
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::end() const -> const_iterator {

    return const_iterator(m_ptr + N);
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::cend() const -> const_iterator {

    return end();
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::rbegin() -> reverse_iterator {

    return reverse_iterator(end());
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::rbegin() const
    -> const_reverse_iterator {

    return const_reverse_iterator(end());
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::crbegin() const
    -> const_reverse_iterator {

    return rbegin();
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::rend() -> reverse_iterator {

    return reverse_iterator(begin());
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::rend() const
    -> const_reverse_iterator {

    return const_reverse_iterator(begin());
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE auto device_array<T, N>::crend() const
    -> const_reverse_iterator {

    return rend();
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr bool device_array<T, N>::empty() const {

    return N == 0;
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr auto device_array<T, N>::size() const
    -> size_type {

    return N;
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr auto device_array<T, N>::max_size() const
    -> size_type {

    return size();
}

}  // namespace vecmem
