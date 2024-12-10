/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <cassert>
#include <cstring>
#include <utility>

namespace vecmem {

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE static_vector<TYPE, MAX_SIZE>::static_vector()
    : m_size(0), m_elements() {}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE static_vector<TYPE, MAX_SIZE>::static_vector(
    size_type size, const_reference value)
    : m_size(size), m_elements() {

    assign(size, value);
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE static_vector<TYPE, MAX_SIZE>::static_vector(
    const static_vector& parent)
    : m_size(parent.m_size), m_elements() {

    // Make copies of all of the elements.
    for (size_type i = 0; i < m_size; ++i) {
        construct(i, parent[i]);
    }
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE static_vector<TYPE, MAX_SIZE>::~static_vector() {

    clear();
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE typename static_vector<TYPE, MAX_SIZE>::reference
static_vector<TYPE, MAX_SIZE>::at(size_type pos) {

    // Make sure that the element exists.
    assert(pos < m_size);

    // Return the element.
    return *(reinterpret_cast<pointer>(m_elements) + pos);
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::at(
    size_type pos) const -> const_reference {

    // Make sure that the element exists.
    assert(pos < m_size);

    // Return the element.
    return *(reinterpret_cast<const_pointer>(m_elements) + pos);
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::operator[](
    size_type pos) -> reference {

    // Return the element.
    return *(reinterpret_cast<pointer>(m_elements) + pos);
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::operator[](
    size_type pos) const -> const_reference {

    // Return the element.
    return *(reinterpret_cast<const_pointer>(m_elements) + pos);
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::front()
    -> reference {

    // Make sure that the element exists.
    assert(m_size > 0);

    // Return the element.
    return *(reinterpret_cast<pointer>(m_elements));
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::front() const
    -> const_reference {

    // Make sure that the element exists.
    assert(m_size > 0);

    // Return the element.
    return *(reinterpret_cast<const_pointer>(m_elements));
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::back() -> reference {

    // Make sure that the element exists.
    assert(m_size > 0);

    // Return the element.
    return *(reinterpret_cast<pointer>(m_elements) + m_size - 1);
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::back() const
    -> const_reference {

    // Make sure that the element exists.
    assert(m_size > 0);

    // Return the element.
    return *(reinterpret_cast<const_pointer>(m_elements) + m_size - 1);
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::data() -> pointer {

    return reinterpret_cast<pointer>(m_elements);
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::data() const
    -> const_pointer {

    return reinterpret_cast<const_pointer>(m_elements);
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE void static_vector<TYPE, MAX_SIZE>::assign(
    size_type count, const_reference value) {

    // Make sure that the sizes are compatible.
    assert(array_max_size >= count);

    // Remove all previous elements.
    clear();

    // Create the required number of identical elements.
    for (size_type i = 0; i < count; ++i) {
        construct(i, value);
    }

    // Set the assigned size of the vector.
    m_size = count;
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::insert(
    const_iterator pos, const_reference value) -> iterator {

    // Make sure that one more position is available.
    assert(m_size < array_max_size);

    // Find the index of this iterator inside of the vector.
    auto id = element_id(pos);

    // Move the payload of the existing elements after "pos".
    memmove(static_cast<void*>(id.m_ptr + 1),
            static_cast<const void*>(id.m_ptr),
            (m_size - id.m_index) * value_size);

    // Instantiate the new value.
    construct(id.m_index, value);

    // Increment the size.
    ++m_size;

    // Return an iterator to the inserted element.
    return id.m_ptr;
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::insert(
    const_iterator pos, size_type count, const_reference value) -> iterator {

    // Make sure that the requested number of positions are still available.
    assert(m_size + count <= array_max_size);

    // Find the index of this iterator inside of the vector.
    auto id = element_id(pos);

    // Move the payload of the existing elements after "pos".
    memmove(static_cast<void*>(id.m_ptr + count),
            static_cast<const void*>(id.m_ptr),
            (m_size - id.m_index) * value_size);

    // Instantiate all the new values.
    for (size_type i = 0; i < count; ++i) {
        construct(id.m_index + i, value);
    }

    // Increment the size.
    m_size += count;

    // Return an iterator to the first inserted element.
    return id.m_ptr;
}

template <typename TYPE, std::size_t MAX_SIZE>
template <typename... Args>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::emplace(
    const_iterator pos, Args&&... args) -> iterator {

    // Make sure that one more position is available.
    assert(m_size < array_max_size);

    // Find the index of this iterator inside of the vector.
    auto id = element_id(pos);

    // Move the payload of the existing elements after "pos".
    memmove(static_cast<void*>(id.m_ptr + 1),
            static_cast<const void*>(id.m_ptr),
            (m_size - id.m_index) * value_size);

    // Instantiate the new value.
    new (id.m_ptr) value_type(std::forward<Args>(args)...);

    // Increment the size.
    ++m_size;

    // Return an iterator to the inserted element.
    return id.m_ptr;
}

template <typename TYPE, std::size_t MAX_SIZE>
template <typename... Args>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::emplace_back(
    Args&&... args) -> reference {

    return *(emplace(end(), std::forward<Args>(args)...));
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE void static_vector<TYPE, MAX_SIZE>::push_back(
    const_reference value) {

    insert(end(), value);
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::erase(
    const_iterator pos) -> iterator {

    // Find the index of this iterator inside of the vector.
    auto id = element_id(pos);

    // Destroy the object.
    destruct(id.m_index);

    // Move up the payload of the elements from after the removed one.
    memmove(static_cast<void*>(id.m_ptr),
            static_cast<const void*>(id.m_ptr + 1),
            (m_size - id.m_index - 1) * value_size);

    // Decrement the size.
    --m_size;

    // Return an iterator to after the removed element.
    return id.m_ptr;
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::erase(
    const_iterator first, const_iterator last) -> iterator {

    // Find the indices and pointers of the iterators.
    auto first_id = element_id(first);
    auto last_id = element_id(last);
    assert(first_id.m_index <= last_id.m_index);

    // Destroy the objects.
    for (size_type i = first_id.m_index; i < last_id.m_index; ++i) {
        destruct(i);
    }

    // Move up the payload of the elements from after the removed range.
    memmove(static_cast<void*>(first_id.m_ptr),
            static_cast<const void*>(last_id.m_ptr),
            (m_size - last_id.m_index) * value_size);

    // Decrease the size.
    m_size -= (last_id.m_index - first_id.m_index);

    // Return an iterator to after the removed elements.
    return first_id.m_ptr;
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE void static_vector<TYPE, MAX_SIZE>::pop_back() {

    erase(end() - 1);
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE void static_vector<TYPE, MAX_SIZE>::clear() {

    for (size_type i = 0; i < m_size; ++i) {
        destruct(i);
    }
    m_size = 0;
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE void static_vector<TYPE, MAX_SIZE>::resize(
    std::size_t new_size) {

    resize(new_size, value_type());
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE void static_vector<TYPE, MAX_SIZE>::resize(
    std::size_t new_size, const_reference value) {

    // Make sure that the request can be done.
    assert(new_size <= array_max_size);

    // Check if anything even needs to be done.
    if (new_size == m_size) {
        return;
    }

    // If the new size is smaller than the current size, remove the unwanted
    // elements.
    if (new_size < m_size) {
        erase(begin() + new_size, end());
    }
    // If the new size is larger than the current size, insert extra elements.
    else {
        insert(end(), new_size - m_size, value);
    }
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::begin() -> iterator {

    return reinterpret_cast<iterator>(m_elements);
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::begin() const
    -> const_iterator {

    return reinterpret_cast<const_iterator>(m_elements);
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::cbegin() const
    -> const_iterator {

    return begin();
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::end() -> iterator {

    return (reinterpret_cast<iterator>(m_elements) + m_size);
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::end() const
    -> const_iterator {

    return (reinterpret_cast<const_iterator>(m_elements) + m_size);
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::cend() const
    -> const_iterator {

    return end();
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::rbegin()
    -> reverse_iterator {

    return reverse_iterator(end());
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::rbegin() const
    -> const_reverse_iterator {

    return const_reverse_iterator(end());
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::crbegin() const
    -> const_reverse_iterator {

    return rbegin();
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::rend()
    -> reverse_iterator {

    return reverse_iterator(begin());
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::rend() const
    -> const_reverse_iterator {

    return const_reverse_iterator(begin());
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::crend() const
    -> const_reverse_iterator {

    return rend();
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE bool static_vector<TYPE, MAX_SIZE>::empty() const {

    return m_size == 0;
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::size() const
    -> size_type {

    return m_size;
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::max_size() const
    -> size_type {

    return array_max_size;
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE auto static_vector<TYPE, MAX_SIZE>::capacity() const
    -> size_type {

    return array_max_size;
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE void static_vector<TYPE, MAX_SIZE>::reserve(
    size_type new_cap) {

    // Make sure that the user didn't ask for too much.
    assert(new_cap <= array_max_size);
    (void)new_cap;
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE void static_vector<TYPE, MAX_SIZE>::construct(
    size_type pos, const_reference value) {

    // Make sure that the position is available.
    assert(pos < array_max_size);

    // Use the constructor of the type.
    pointer ptr = reinterpret_cast<pointer>(m_elements) + pos;
    new (ptr) value_type(value);
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE void static_vector<TYPE, MAX_SIZE>::destruct(
    size_type pos) {

    // Make sure that the element has been allocated.
    assert(pos < m_size);

    // Use the destructor of the type.
    pointer ptr = reinterpret_cast<pointer>(m_elements) + pos;
    ptr->~value_type();
}

template <typename TYPE, std::size_t MAX_SIZE>
VECMEM_HOST_AND_DEVICE typename static_vector<TYPE, MAX_SIZE>::ElementId
static_vector<TYPE, MAX_SIZE>::element_id(const_iterator pos) {

    size_type const index = static_cast<size_type>(pos - begin());
    assert(index <= m_size);
    pointer const ptr = reinterpret_cast<pointer>(m_elements) + index;
    return {index, ptr};
}

}  // namespace vecmem
