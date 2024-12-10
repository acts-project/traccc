/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

#include "vecmem/memory/unique_ptr.hpp"

// System include(s).
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

namespace vecmem {
template <typename T, std::size_t N>
array<T, N>::array(memory_resource& resource)
    : m_size(N), m_memory(vecmem::make_unique_obj<T[]>(resource, m_size)) {

    static_assert(N != details::array_invalid_size,
                  "Can only use the 'compile time constructor' if a size "
                  "was provided as a template argument");
}

template <typename T, std::size_t N>
array<T, N>::array(memory_resource& resource, size_type size)
    : m_size(size), m_memory(vecmem::make_unique_obj<T[]>(resource, m_size)) {

    static_assert(N == details::array_invalid_size,
                  "Can only use the 'runtime constructor' if a size was not "
                  "provided as a template argument");
}

template <typename T, std::size_t N>
auto array<T, N>::at(size_type pos) -> reference {

    if (pos >= m_size) {
        throw std::out_of_range("Requested element " + std::to_string(pos) +
                                " from a " + std::to_string(m_size) +
                                " sized vecmem::array");
    }
    return m_memory.get()[pos];
}

template <typename T, std::size_t N>
auto array<T, N>::at(size_type pos) const -> const_reference {

    if (pos >= m_size) {
        throw std::out_of_range("Requested element " + std::to_string(pos) +
                                " from a " + std::to_string(m_size) +
                                " sized vecmem::array");
    }
    return m_memory.get()[pos];
}

template <typename T, std::size_t N>
auto array<T, N>::operator[](size_type pos) -> reference {

    return m_memory.get()[pos];
}

template <typename T, std::size_t N>
auto array<T, N>::operator[](size_type pos) const -> const_reference {

    return m_memory.get()[pos];
}

template <typename T, std::size_t N>
auto array<T, N>::front() -> reference {

    if (m_size == 0) {
        throw std::out_of_range(
            "Called vecmem::array::front() on an empty "
            "array");
    }
    return (m_memory[0]);
}

template <typename T, std::size_t N>
auto array<T, N>::front() const -> const_reference {

    if (m_size == 0) {
        throw std::out_of_range(
            "Called vecmem::array::front() on an empty "
            "array");
    }
    return (*m_memory);
}

template <typename T, std::size_t N>
auto array<T, N>::back() -> reference {

    if (m_size == 0) {
        throw std::out_of_range(
            "Called vecmem::array::back() on an empty "
            "array");
    }
    return m_memory.get()[m_size - 1];
}

template <typename T, std::size_t N>
auto array<T, N>::back() const -> const_reference {

    if (m_size == 0) {
        throw std::out_of_range(
            "Called vecmem::array::back() on an empty "
            "array");
    }
    return m_memory.get()[m_size - 1];
}

template <typename T, std::size_t N>
auto array<T, N>::data() -> pointer {

    return m_memory.get();
}

template <typename T, std::size_t N>
auto array<T, N>::data() const -> const_pointer {

    return m_memory.get();
}

template <typename T, std::size_t N>
auto array<T, N>::begin() -> iterator {

    return m_memory.get();
}

template <typename T, std::size_t N>
auto array<T, N>::begin() const -> const_iterator {

    return m_memory.get();
}

template <typename T, std::size_t N>
auto array<T, N>::cbegin() const -> const_iterator {

    return m_memory.get();
}

template <typename T, std::size_t N>
auto array<T, N>::end() -> iterator {

    return (m_memory.get() + m_size);
}

template <typename T, std::size_t N>
auto array<T, N>::end() const -> const_iterator {

    return (m_memory.get() + m_size);
}

template <typename T, std::size_t N>
auto array<T, N>::cend() const -> const_iterator {

    return (m_memory.get() + m_size);
}

template <typename T, std::size_t N>
auto array<T, N>::rbegin() -> reverse_iterator {

    return reverse_iterator(end());
}

template <typename T, std::size_t N>
auto array<T, N>::rbegin() const -> const_reverse_iterator {

    return const_reverse_iterator(end());
}

template <typename T, std::size_t N>
auto array<T, N>::crbegin() const -> const_reverse_iterator {

    return const_reverse_iterator(end());
}

template <typename T, std::size_t N>
auto array<T, N>::rend() -> reverse_iterator {

    return reverse_iterator(begin());
}

template <typename T, std::size_t N>
auto array<T, N>::rend() const -> const_reverse_iterator {

    return const_reverse_iterator(begin());
}

template <typename T, std::size_t N>
auto array<T, N>::crend() const -> const_reverse_iterator {

    return const_reverse_iterator(begin());
}

template <typename T, std::size_t N>
bool array<T, N>::empty() const noexcept {

    return (m_size == 0);
}

template <typename T, std::size_t N>
auto array<T, N>::size() const noexcept -> size_type {

    return m_size;
}

template <typename T, std::size_t N>
void array<T, N>::fill(const_reference value) {

    std::fill(begin(), end(), value);
}

template <typename T, std::size_t N>
VECMEM_HOST data::vector_view<T> get_data(array<T, N>& a) {

    return {static_cast<typename data::vector_view<T>::size_type>(a.size()),
            a.data()};
}

template <typename T, std::size_t N>
VECMEM_HOST data::vector_view<const T> get_data(const array<T, N>& a) {

    return {
        static_cast<typename data::vector_view<const T>::size_type>(a.size()),
        a.data()};
}

}  // namespace vecmem
