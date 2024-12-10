/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// vecmem include(s).
#include "vecmem/containers/details/aligned_multiple_placement.hpp"

// System include(s).
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <numeric>
#include <vector>

namespace {

/// Helper conversion function
template <typename TYPE>
std::vector<std::size_t> get_sizes(
    const vecmem::data::jagged_vector_view<TYPE>& jvv) {

    std::vector<std::size_t> result(jvv.size());
    std::transform(jvv.host_ptr(), jvv.host_ptr() + jvv.size(), result.begin(),
                   [](const auto& vv) { return vv.size(); });
    return result;
}

/// Function allocating memory for @c vecmem::data::jagged_vector_buffer
template <typename TYPE>
vecmem::unique_alloc_ptr<
    typename vecmem::data::jagged_vector_buffer<TYPE>::value_type[]>
allocate_jagged_buffer_outer_memory(
    typename vecmem::data::jagged_vector_buffer<TYPE>::size_type size,
    vecmem::memory_resource& resource) {

    if (size == 0) {
        return nullptr;
    } else {
        return vecmem::make_unique_alloc<
            typename vecmem::data::jagged_vector_buffer<TYPE>::value_type[]>(
            resource, size);
    }
}
}  // namespace

namespace vecmem {
namespace data {

/// A custom implementation for the default constructor is necessary because
/// @c vecmem::data::jagged_vector_view does not set its members to anything
/// explicitly in its default constructor. (In order to be trivially default
/// constructible.) So here we need to be explicit.
template <typename TYPE>
jagged_vector_buffer<TYPE>::jagged_vector_buffer() : base_type(0, nullptr) {}

template <typename TYPE>
template <typename OTHERTYPE,
          std::enable_if_t<std::is_convertible<TYPE, OTHERTYPE>::value, bool> >
jagged_vector_buffer<TYPE>::jagged_vector_buffer(
    const jagged_vector_view<OTHERTYPE>& other, memory_resource& resource,
    memory_resource* host_access_resource, buffer_type type)
    : jagged_vector_buffer(::get_sizes(other), resource, host_access_resource,
                           type) {}

template <typename TYPE>
template <typename SIZE_TYPE,
          std::enable_if_t<std::is_integral<SIZE_TYPE>::value &&
                               std::is_unsigned<SIZE_TYPE>::value,
                           bool> >
jagged_vector_buffer<TYPE>::jagged_vector_buffer(
    const std::vector<SIZE_TYPE>& capacities, memory_resource& resource,
    memory_resource* host_access_resource, buffer_type type)
    : base_type(capacities.size(), nullptr),
      m_outer_memory(::allocate_jagged_buffer_outer_memory<TYPE>(
          (host_access_resource == nullptr ? 0 : capacities.size()), resource)),
      m_outer_host_memory(::allocate_jagged_buffer_outer_memory<TYPE>(
          capacities.size(),
          (host_access_resource == nullptr ? resource
                                           : *host_access_resource))) {

    // Determine the allocation size.
    using header_t = typename vecmem::data::jagged_vector_buffer<
        TYPE>::value_type::size_type;
    const std::size_t total_elements = std::accumulate(
        capacities.begin(), capacities.end(), static_cast<std::size_t>(0));

    // Helper pointers to the "inner data".
    header_t* header_ptr = nullptr;
    TYPE* data_ptr = nullptr;

    // Allocate the "inner memory" for a fixed size buffer.
    if (type == buffer_type::fixed_size && total_elements != 0) {
        m_inner_memory = vecmem::make_unique_alloc<char[]>(
            resource, total_elements * sizeof(TYPE));
        data_ptr = reinterpret_cast<TYPE*>(m_inner_memory.get());
    }
    // Allocate the "inner memory" for a resizable buffer.
    else if (type == buffer_type::resizable && capacities.size() != 0) {
        std::tie(m_inner_memory, header_ptr, data_ptr) =
            details::aligned_multiple_placement<header_t, TYPE>(
                resource, capacities.size(), total_elements);
    }

    // Point the base class at the newly allocated memory.
    base_type::m_ptr =
        ((host_access_resource != nullptr) ? m_outer_memory.get()
                                           : m_outer_host_memory.get());
    base_type::m_host_ptr = m_outer_host_memory.get();

    // Set up the vecmem::vector_view objects in the host accessible memory.
    std::ptrdiff_t ptrdiff = 0;
    for (std::size_t i = 0; i < capacities.size(); ++i) {
        if (header_ptr != nullptr) {
            new (base_type::host_ptr() + i) value_type(
                static_cast<typename value_type::size_type>(capacities[i]),
                &(header_ptr[i]), data_ptr + ptrdiff);
        } else {
            new (base_type::host_ptr() + i) value_type(
                static_cast<typename value_type::size_type>(capacities[i]),
                data_ptr + ptrdiff);
        }
        ptrdiff += capacities[i];
    }
}

}  // namespace data

template <typename TYPE>
data::jagged_vector_view<TYPE>& get_data(
    data::jagged_vector_buffer<TYPE>& data) {

    return data;
}

template <typename TYPE>
const data::jagged_vector_view<TYPE>& get_data(
    const data::jagged_vector_buffer<TYPE>& data) {

    return data;
}

}  // namespace vecmem
