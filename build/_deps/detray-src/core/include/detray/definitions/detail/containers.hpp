/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/utils/tuple.hpp"

// Vecmem include(s)
#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/containers/vector.hpp>

// System include(s)
#include <array>
#include <map>
#include <type_traits>
#include <vector>

namespace detray {

template <typename value_t, std::size_t kDIM>
using darray = std::array<value_t, kDIM>;

template <typename value_t>
using dvector = vecmem::vector<value_t>;

template <typename value_t>
using djagged_vector = vecmem::jagged_vector<value_t>;

template <typename key_t, typename value_t>
using dmap = std::map<key_t, value_t>;

template <class... types>
using dtuple = detray::tuple<types...>;

/// @brief Bundle container type definitions
template <template <typename...> class vector_t = dvector,
          template <typename...> class tuple_t = dtuple,
          template <typename, std::size_t> class array_t = darray,
          template <typename...> class jagged_vector_t = djagged_vector,
          template <typename, typename> class map_t = dmap>
struct container_types {
    template <typename T>
    using vector_type = vector_t<T>;

    template <class... T>
    using tuple_type = tuple_t<T...>;

    template <typename T, std::size_t kDIM>
    using array_type = array_t<T, kDIM>;

    template <typename T>
    using jagged_vector_type = jagged_vector_t<T>;

    template <typename K, typename T>
    using map_type = map_t<K, T>;
};

/// Defining some common types
using host_container_types = container_types<>;

namespace detail {

// make std::get available in detray detail namespace, where also the thrust and
// index specific overloads live.
using std::get;

/// Trait class to figure out if a given type has a @c reserve(...) function
template <typename T>
struct has_reserve {

    private:
    /// Function returning @c std::true_type for types that do have a @c
    /// reserve(...) function
    template <typename C>
    static constexpr auto check(C*) ->
        typename std::is_void<decltype(std::declval<C>().reserve(
            std::declval<typename C::size_type>()))>::type;

    /// Function returning @c std::false_type for types that fair the previous
    /// function
    template <typename>
    static constexpr std::false_type check(...);

    /// Declare the value type of this trait class
    using type = decltype(check<T>(nullptr));

    public:
    /// Value of the check
    static constexpr bool value = type::value;
};

/// @name Functions calling or not calling reserve(...) based on whether it's
/// available
/// @{
template <typename T>
requires has_reserve<T>::value DETRAY_HOST_DEVICE void call_reserve(
    T& obj, std::size_t newsize) {
    obj.reserve(newsize);
}

template <typename T>
requires(!has_reserve<T>::value) DETRAY_HOST_DEVICE
    void call_reserve(T& /*obj*/, std::size_t /*newsize*/) {
    /*Not defined*/
}
/// @}

}  // namespace detail

}  // namespace detray
