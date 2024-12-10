/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/jagged_vector_data.hpp"
#include "vecmem/edm/details/schema_traits.hpp"
#include "vecmem/edm/details/view_traits.hpp"
#include "vecmem/edm/schema.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/utils/tuple.hpp"

// System include(s).
#include <tuple>
#include <variant>

namespace vecmem::edm::details {

/// @name Traits for the data types for the individual variables
/// @{

template <typename TYPE>
struct data_type {
    using type = std::monostate;
};  // struct data_type

template <typename TYPE>
struct data_type<type::jagged_vector<TYPE>> {
    using type = vecmem::data::jagged_vector_data<TYPE>;
};  // struct data_type

/// @}

/// @name Traits for allocating data variables
/// @{

template <typename TYPE>
struct data_alloc {
    static typename data_type<TYPE>::type make(std::size_t, memory_resource&) {
        return {};
    }
};  // struct data_alloc

template <typename TYPE>
struct data_alloc<type::jagged_vector<TYPE>> {
    static typename data_type<type::jagged_vector<TYPE>>::type make(
        std::size_t size, memory_resource& mr) {
        return {size, mr};
    }
};  // struct data_alloc

/// @}

/// Helper function assigning variable data objects to view objects
template <typename... VARTYPES, std::size_t INDEX, std::size_t... Is>
void data_view_assign(
    tuple<typename view_type<VARTYPES>::type...>& view,
    const std::tuple<typename data_type<VARTYPES>::type...>& data,
    std::index_sequence<INDEX, Is...>) {

    // Make the assignments just for jagged vector variables.
    if constexpr (type::details::is_jagged_vector<typename std::tuple_element<
                      INDEX, std::tuple<VARTYPES...>>::type>::value) {
        get<INDEX>(view) = std::get<INDEX>(data);
    }
    // Terminate, or continue.
    if constexpr (sizeof...(Is) > 0) {
        data_view_assign<VARTYPES...>(view, data, std::index_sequence<Is...>{});
    }
}

}  // namespace vecmem::edm::details
