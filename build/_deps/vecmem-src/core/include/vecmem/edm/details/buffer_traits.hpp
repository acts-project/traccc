/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/details/schema_traits.hpp"
#include "vecmem/edm/details/view_traits.hpp"
#include "vecmem/edm/schema.hpp"
#include "vecmem/edm/view.hpp"
#include "vecmem/memory/unique_ptr.hpp"
#include "vecmem/utils/tuple.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <array>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

namespace vecmem::edm::details {

/// @name Traits used for making allocations inside of buffers
/// @{

template <typename TYPE>
struct buffer_alloc;

template <typename TYPE>
struct buffer_alloc<type::scalar<TYPE> > {
    /// The number of @c TYPE elements to allocate for the payload
    template <typename SIZE_TYPE = std::size_t>
    VECMEM_HOST static std::size_t payload_size(const std::vector<SIZE_TYPE>&) {
        return 1u;
    }
    /// The number of "layout meta-objects" to allocate for the payload
    template <typename SIZE_TYPE = std::size_t>
    VECMEM_HOST static std::size_t layout_size(const std::vector<SIZE_TYPE>&) {
        return 0u;
    }
    /// Construct a view for a scalar variable.
    template <typename SIZE_TYPE = std::size_t>
    VECMEM_HOST static typename view_type<type::scalar<TYPE> >::type make_view(
        const std::vector<SIZE_TYPE>&, unsigned int*,
        typename view_type<type::scalar<TYPE> >::layout_ptr,
        typename view_type<type::scalar<TYPE> >::layout_ptr,
        typename view_type<type::scalar<TYPE> >::payload_ptr ptr) {
        return ptr;
    }
};  // struct buffer_alloc

template <typename TYPE>
struct buffer_alloc<type::vector<TYPE> > {
    /// The number of @c TYPE elements to allocate for the payload
    template <typename SIZE_TYPE = std::size_t>
    VECMEM_HOST static std::size_t payload_size(
        const std::vector<SIZE_TYPE>& sizes) {
        return sizes.size();
    }
    /// The number of "layout meta-objects" to allocate for the payload
    template <typename SIZE_TYPE = std::size_t>
    VECMEM_HOST static std::size_t layout_size(const std::vector<SIZE_TYPE>&) {
        return 0u;
    }
    /// Construct a view for a vector variable.
    template <typename SIZE_TYPE = std::size_t>
    VECMEM_HOST static typename view_type<type::vector<TYPE> >::type make_view(
        const std::vector<SIZE_TYPE>& capacity, unsigned int* size,
        typename view_type<type::vector<TYPE> >::layout_ptr,
        typename view_type<type::vector<TYPE> >::layout_ptr,
        typename view_type<type::vector<TYPE> >::payload_ptr ptr) {
        return {static_cast<
                    typename view_type<type::vector<TYPE> >::type::size_type>(
                    capacity.size()),
                size, ptr};
    }
};  // struct buffer_alloc

template <typename TYPE>
struct buffer_alloc<type::jagged_vector<TYPE> > {
    /// The number of @c TYPE elements to allocate for the payload
    template <typename SIZE_TYPE = std::size_t>
    VECMEM_HOST static std::size_t payload_size(
        const std::vector<SIZE_TYPE>& sizes) {
        return std::accumulate(sizes.begin(), sizes.end(),
                               static_cast<std::size_t>(0));
    }
    /// The number of "layout meta-objects" to allocate for the payload
    template <typename SIZE_TYPE = std::size_t>
    VECMEM_HOST static std::size_t layout_size(
        const std::vector<SIZE_TYPE>& sizes) {
        return sizes.size();
    }
    /// Construct a view for a jagged vector variable.
    template <typename SIZE_TYPE = std::size_t>
    VECMEM_HOST static typename view_type<type::jagged_vector<TYPE> >::type
    make_view(
        const std::vector<SIZE_TYPE>& capacities, unsigned int* sizes,
        typename view_type<type::jagged_vector<TYPE> >::layout_ptr layout,
        typename view_type<type::jagged_vector<TYPE> >::layout_ptr host_layout,
        typename view_type<type::jagged_vector<TYPE> >::payload_ptr ptr) {

        // Set up the "layout objects" for use by the view.
        typename view_type<type::jagged_vector<TYPE> >::layout_ptr used_layout =
            (host_layout != nullptr ? host_layout : layout);
        std::ptrdiff_t ptrdiff = 0;
        for (std::size_t i = 0; i < capacities.size(); ++i) {
            new (used_layout + i)
                typename view_type<type::jagged_vector<TYPE> >::layout_type(
                    static_cast<typename view_type<
                        type::jagged_vector<TYPE> >::layout_type::size_type>(
                        capacities[i]),
                    (sizes != nullptr ? &(sizes[i]) : nullptr), ptr + ptrdiff);
            ptrdiff += capacities[i];
        }

        // Create the jagged vector view.
        return {static_cast<
                    typename view_type<type::vector<TYPE> >::type::size_type>(
                    capacities.size()),
                layout, host_layout};
    }
};  // struct buffer_alloc

/// @}

/// Function constructing fixed size view objects for @c vecmem::edm::buffer
template <typename SIZE_TYPE, typename... TYPES, std::size_t... INDICES>
VECMEM_HOST auto make_buffer_views(
    const std::vector<SIZE_TYPE>& sizes,
    const std::tuple<typename view_type<TYPES>::layout_ptr...>& layouts,
    const std::tuple<typename view_type<TYPES>::layout_ptr...>& host_layouts,
    const std::tuple<typename view_type<TYPES>::payload_ptr...>& payloads,
    std::index_sequence<INDICES...>) {

    return vecmem::make_tuple(buffer_alloc<TYPES>::make_view(
        sizes, nullptr, std::get<INDICES>(layouts),
        std::get<INDICES>(host_layouts), std::get<INDICES>(payloads))...);
}

/// Function constructing resizable view objects for @c vecmem::edm::buffer
template <typename SIZE_TYPE, typename... TYPES, std::size_t... INDICES>
VECMEM_HOST auto make_buffer_views(
    const std::vector<SIZE_TYPE>& capacities,
    const std::tuple<typename view_type<TYPES>::size_ptr...>& sizes,
    const std::tuple<typename view_type<TYPES>::layout_ptr...>& layouts,
    const std::tuple<typename view_type<TYPES>::layout_ptr...>& host_layouts,
    const std::tuple<typename view_type<TYPES>::payload_ptr...>& payloads,
    const std::index_sequence<INDICES...>) {

    // Helper variable(s).
    constexpr auto is_jagged_vector =
        std::make_tuple(type::details::is_jagged_vector<TYPES>::value...);
    constexpr bool has_jagged_vector =
        std::disjunction_v<type::details::is_jagged_vector<TYPES>...>;

    // The logic here is that if there are any jagged vectors in the schema,
    // then only the jagged vectors are resizable, the "normal vectors" are not.
    // But the received "sizes" variable would be hard to set up like that
    // outside of this function, so the logic has to sit here.
    return vecmem::make_tuple(buffer_alloc<TYPES>::make_view(
        capacities,
        ((has_jagged_vector && (!std::get<INDICES>(is_jagged_vector)))
             ? nullptr
             : std::get<INDICES>(sizes)),
        std::get<INDICES>(layouts), std::get<INDICES>(host_layouts),
        std::get<INDICES>(payloads))...);
}

/// Generic function finding the first non-nullptr pointer in a tuple.
template <typename... TYPES, std::size_t INDEX, std::size_t... INDICES>
VECMEM_HOST constexpr void* find_first_pointer(
    const std::tuple<TYPES...>& pointers,
    std::index_sequence<INDEX, INDICES...>) {

    auto ptr = std::get<INDEX>(pointers);
    if (ptr != nullptr) {
        return ptr;
    } else {
        if constexpr (sizeof...(INDICES) > 0) {
            return find_first_pointer<TYPES...>(
                pointers, std::index_sequence<INDICES...>());
        } else {
            return nullptr;
        }
    }
}

/// Generic function finding the last non-nullptr pointer in a tuple.
template <typename... TYPES, std::size_t INDEX, std::size_t... INDICES>
VECMEM_HOST constexpr void* find_last_pointer(
    const std::tuple<TYPES...>& pointers,
    const std::array<std::size_t, sizeof...(TYPES)>& sizes,
    std::index_sequence<INDEX, INDICES...>) {

    auto ptr = std::get<sizeof...(TYPES) - 1 - INDEX>(pointers);
    if (ptr != nullptr) {
        return ptr + std::get<sizeof...(TYPES) - 1 - INDEX>(sizes);
    } else {
        if constexpr (sizeof...(INDICES) > 0) {
            return find_last_pointer<TYPES...>(
                pointers, sizes, std::index_sequence<INDICES...>());
        } else {
            return nullptr;
        }
    }
}

/// Function creating a view for the layout of a buffer.
template <typename... TYPES>
VECMEM_HOST constexpr typename view<schema<TYPES...> >::memory_view_type
find_layout_view(
    const std::tuple<typename view_type<TYPES>::layout_ptr...>& layouts,
    const std::array<std::size_t, sizeof...(TYPES)>& sizes) {

    // The result type.
    using result_type = typename view<schema<TYPES...> >::memory_view_type;

    // Find the first non-zero pointer.
    typename result_type::pointer ptr =
        reinterpret_cast<typename result_type::pointer>(
            find_first_pointer(layouts, std::index_sequence_for<TYPES...>()));
    // Find the last non-zero pointer.
    typename result_type::pointer end_ptr =
        reinterpret_cast<typename result_type::pointer>(find_last_pointer(
            layouts, sizes, std::index_sequence_for<TYPES...>()));

    // Construct the result.
    return {static_cast<typename result_type::size_type>(end_ptr - ptr), ptr};
}

/// Function creating a view of the payload of a buffer
template <typename... TYPES>
VECMEM_HOST constexpr typename view<schema<TYPES...> >::memory_view_type
find_payload_view(
    const std::tuple<typename view_type<TYPES>::payload_ptr...>& payloads,
    const std::array<std::size_t, sizeof...(TYPES)>& sizes) {

    // The result type.
    using result_type = typename view<schema<TYPES...> >::memory_view_type;

    // Find the first non-zero pointer.
    typename result_type::pointer ptr =
        reinterpret_cast<typename result_type::pointer>(
            find_first_pointer(payloads, std::index_sequence_for<TYPES...>()));
    // Find the last non-zero pointer.
    typename result_type::pointer end_ptr =
        reinterpret_cast<typename result_type::pointer>(find_last_pointer(
            payloads, sizes, std::index_sequence_for<TYPES...>()));

    // Construct the result.
    return {static_cast<typename result_type::size_type>(end_ptr - ptr), ptr};
}

}  // namespace vecmem::edm::details
