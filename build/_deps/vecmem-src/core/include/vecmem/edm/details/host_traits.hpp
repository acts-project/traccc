/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/details/resize_jagged_vector.hpp"
#include "vecmem/containers/jagged_vector.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/edm/details/schema_traits.hpp"
#include "vecmem/edm/schema.hpp"
#include "vecmem/memory/memory_resource.hpp"

// System include(s).
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace vecmem::edm::details {

/// @name Traits for the host types for the individual variables
/// @{

template <typename TYPE>
struct host_type;

template <typename TYPE>
struct host_type<type::scalar<TYPE>> {
    using type = vector<TYPE>;
    using return_type = std::add_lvalue_reference_t<TYPE>;
    using const_return_type =
        std::add_lvalue_reference_t<std::add_const_t<TYPE>>;
};  // struct host_type

template <typename TYPE>
struct host_type<type::vector<TYPE>> {
    using type = vector<TYPE>;
    using return_type = std::add_lvalue_reference_t<type>;
    using const_return_type =
        std::add_lvalue_reference_t<std::add_const_t<type>>;
};  // struct host_type

template <typename TYPE>
struct host_type<type::jagged_vector<TYPE>> {
    using type = jagged_vector<TYPE>;
    using return_type = std::add_lvalue_reference_t<type>;
    using const_return_type =
        std::add_lvalue_reference_t<std::add_const_t<type>>;
};  // struct host_type

template <std::size_t INDEX, typename... VARTYPES>
struct host_type_at {
    using type = typename host_type<typename std::tuple_element<
        INDEX, std::tuple<VARTYPES...>>::type>::type;
    using return_type = typename host_type<typename std::tuple_element<
        INDEX, std::tuple<VARTYPES...>>::type>::return_type;
    using const_return_type = typename host_type<typename std::tuple_element<
        INDEX, std::tuple<VARTYPES...>>::type>::const_return_type;
};  // struct host_type_at

/// @}

/// @name Traits for allocating host variables
/// @{

template <typename TYPE>
struct host_alloc {
    static typename host_type<TYPE>::type make(memory_resource& mr) {
        return typename host_type<TYPE>::type{&mr};
    }
};  // struct host_alloc

template <typename TYPE>
struct host_alloc<type::scalar<TYPE>> {
    static typename host_type<type::scalar<TYPE>>::type make(
        memory_resource& mr) {
        return typename host_type<type::scalar<TYPE>>::type{1, &mr};
    }
};  // struct host_alloc

/// @}

/// Recursive function getting the size of a host container
///
/// Note that before calling this function, there is a check that at least one
/// of the variables is a (jagged) vector type. So the index sequence must
/// always contain at least a single element when this function is first called.
///
template <typename... VARTYPES, std::size_t INDEX, std::size_t... Is>
std::size_t get_host_size(
    const std::tuple<typename host_type<VARTYPES>::type...>& data,
    std::index_sequence<INDEX, Is...>, std::size_t size = 0,
    bool size_known = false) {

    // Get the size of this variable.
    std::size_t var_size = 0;
    bool var_size_known = false;
    if constexpr (type::details::is_vector<typename std::tuple_element<
                      INDEX, std::tuple<VARTYPES...>>::type>::value) {
        var_size = std::get<INDEX>(data).size();
        var_size_known = true;
    } else {
        var_size = size;
    }
    // Make sure that it's the same as what has been found before.
    if (size_known && var_size_known && (var_size != size)) {
        throw std::length_error(
            "Inconsistent variable sizes in host container!");
    }
    // Terminate, or continue.
    if constexpr (sizeof...(Is) == 0) {
        if (!(size_known || var_size_known)) {
            throw std::length_error(
                "Could not determine the size of the host container?!?");
        }
        return var_size;
    } else {
        return get_host_size<VARTYPES...>(data, std::index_sequence<Is...>{},
                                          var_size,
                                          size_known || var_size_known);
    }
}

/// Recursive function resizing a host vector
///
/// Note that before calling this function, there is a check that at least one
/// of the variables is a (jagged) vector type. So the index sequence must
/// always contain at least a single element when this function is first called.
///
template <typename... VARTYPES, std::size_t INDEX, std::size_t... Is>
void host_resize(std::tuple<typename host_type<VARTYPES>::type...>& data,
                 std::size_t size, std::index_sequence<INDEX, Is...>) {

    // Resize this variable.
    if constexpr (type::details::is_jagged_vector_v<typename std::tuple_element<
                      INDEX, std::tuple<VARTYPES...>>::type>) {
        vecmem::details::resize_jagged_vector(std::get<INDEX>(data), size);
    } else if constexpr (type::details::is_vector_v<typename std::tuple_element<
                             INDEX, std::tuple<VARTYPES...>>::type>) {
        std::get<INDEX>(data).resize(size);
    }
    // Terminate, or continue.
    if constexpr (sizeof...(Is) > 0) {
        host_resize<VARTYPES...>(data, size, std::index_sequence<Is...>{});
    }
}

/// Recursive function reserving memory for/in a host vector
///
/// Note that before calling this function, there is a check that at least one
/// of the variables is a (jagged) vector type. So the index sequence must
/// always contain at least a single element when this function is first called.
///
template <typename... VARTYPES, std::size_t INDEX, std::size_t... Is>
void host_reserve(std::tuple<typename host_type<VARTYPES>::type...>& data,
                  std::size_t size, std::index_sequence<INDEX, Is...>) {

    // Resize this variable.
    if constexpr (type::details::is_vector<typename std::tuple_element<
                      INDEX, std::tuple<VARTYPES...>>::type>::value) {
        std::get<INDEX>(data).reserve(size);
    }
    // Terminate, or continue.
    if constexpr (sizeof...(Is) > 0) {
        host_reserve<VARTYPES...>(data, size, std::index_sequence<Is...>{});
    }
}

}  // namespace vecmem::edm::details
