/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/details/device_container.hpp"
#include "traccc/edm/details/host_container.hpp"
#include "traccc/utils/type_traits.hpp"

// VecMem include(s).
#include <vecmem/containers/data/jagged_vector_buffer.hpp>
#include <vecmem/containers/data/jagged_vector_data.hpp>
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/vector.hpp>

// System include(s).
#include <type_traits>

namespace traccc {

/// @name Types used to send data back and forth between host and device code
/// @{

/// Structure holding (some of the) data about the container in host code
template <typename header_t, typename item_t>
struct container_data {
    using header_vector = vecmem::data::vector_view<header_t>;
    using item_vector = vecmem::data::jagged_vector_data<item_t>;
    header_vector headers;
    item_vector items;
};

/// Structure holding (all of the) data about the container in host code
template <typename header_t, typename item_t>
struct container_buffer {
    using header_vector = vecmem::data::vector_buffer<header_t>;
    using item_vector = vecmem::data::jagged_vector_buffer<item_t>;
    header_vector headers;
    item_vector items;
};

/// Structure used to send the data about the container to device code
///
/// This is the type that can be passed to device code as-is. But since in
/// host code one needs to manage the data describing a
/// @c traccc::container either using @c traccc::container_data or
/// @c traccc::container_buffer, it needs to have constructors from
/// both of those types.
///
/// In fact it needs to be created from one of those types, as such an
/// object can only function if an instance of one of those types exists
/// alongside it as well.
///
template <typename header_t, typename item_t>
struct container_view {

    /// Type for the header vector (view)
    using header_vector = vecmem::data::vector_view<header_t>;
    /// Type for the item vector (view)
    using item_vector = vecmem::data::jagged_vector_view<item_t>;

    /// Constructor from a @c container_data object
    template <
        typename other_header_t, typename other_item_t,
        std::enable_if_t<details::is_same_nc<header_t, other_header_t>::value,
                         bool> = true,
        std::enable_if_t<details::is_same_nc<item_t, other_item_t>::value,
                         bool> = true>
    TRACCC_HOST_DEVICE container_view(
        const container_data<other_header_t, other_item_t>& data)
        : headers(data.headers), items(data.items) {}

    /// Constructor from a @c container_buffer object
    template <
        typename other_header_t, typename other_item_t,
        std::enable_if_t<details::is_same_nc<header_t, other_header_t>::value,
                         bool> = true,
        std::enable_if_t<details::is_same_nc<item_t, other_item_t>::value,
                         bool> = true>
    TRACCC_HOST_DEVICE container_view(
        const container_buffer<other_header_t, other_item_t>& buffer)
        : headers(buffer.headers), items(buffer.items) {}

    /// Constructor from a non-const view
    template <
        typename other_header_t, typename other_item_t,
        std::enable_if_t<details::is_same_nc<header_t, other_header_t>::value,
                         bool> = true,
        std::enable_if_t<details::is_same_nc<item_t, other_item_t>::value,
                         bool> = true>
    TRACCC_HOST_DEVICE container_view(
        const container_view<other_header_t, other_item_t>& parent)
        : headers(parent.headers), items(parent.items) {}

    /// View of the data describing the headers
    header_vector headers;

    /// View of the data describing the items
    item_vector items;
};

/// Helper function for making a "simple" object out of the container
/// (non-const)
template <typename header_t, typename item_t>
inline container_data<header_t, item_t> get_data(
    host_container<header_t, item_t>& cc,
    vecmem::memory_resource* resource = nullptr) {
    return {{vecmem::get_data(cc.get_headers())},
            {vecmem::get_data(cc.get_items(), resource)}};
}

/// Helper function for making a "simple" object out of the container (const)
template <typename header_t, typename item_t>
inline container_data<const header_t, const item_t> get_data(
    const host_container<header_t, item_t>& cc,
    vecmem::memory_resource* resource = nullptr) {
    return {{vecmem::get_data(cc.get_headers())},
            {vecmem::get_data(cc.get_items(), resource)}};
}

}  // namespace traccc

/// Helper macro declaring all "collection types" for a given type
///
/// Usage:
///
///  namespace foo {
///     struct bar {};
///     TRACCC_DECLARE_COLLECTION_TYPES(bar);
///  }
///
/// In this example this would create all of the following types:
///   - foo::bar_collection<T>
///   - foo::bar_const_collection<T>
///   - foo::host_bar_collection
///   - foo::device_bar_collection
///   - foo::device_bar_const_collection
///
#define TRACCC_DECLARE_COLLECTION_TYPES(TYPE)                           \
    template <template <typename> class vector_t>                       \
    using TYPE##_collection = vector_t<TYPE>;                           \
    using host_##TYPE##_collection = TYPE##_collection<vecmem::vector>; \
    using device_##TYPE##_collection =                                  \
        TYPE##_collection<vecmem::device_vector>;                       \
    template <template <typename> class vector_t>                       \
    using TYPE##_const_collection = vector_t<const TYPE>;               \
    using device_##TYPE##_const_collection =                            \
        TYPE##_const_collection<vecmem::device_vector>

/// Helper macro declaring all container types for a header and item definition
///
/// Usage:
///
///  namespace foo {
///     struct bar_header {};
///     struct bar {};
///     TRACCC_DECLARE_CONTAINER_TYPES(bar, bar_header, bar);
///  }
///
/// In this example it would create all of the following types:
///   - foo::bar_collection<T>
///   - foo::bar_const_collection<T>
///   - foo::host_bar_collection
///   - foo::device_bar_collection
///   - foo::device_bar_const_collection
///   - foo::host_bar_container
///   - foo::device_bar_container
///   - foo::device_bar_const_container
///   - foo::bar_container_view
///   - foo::bar_container_const_view
///   - foo::bar_container_data
///   - foo::bar_container_const_data
///   - foo::bar_container_buffer
///
#define TRACCC_DECLARE_CONTAINER_TYPES(NAME, HEADER_TYPE, ITEM_TYPE)  \
    using host_##NAME##_container =                                   \
        traccc::host_container<HEADER_TYPE, ITEM_TYPE>;               \
    using device_##NAME##_container =                                 \
        traccc::device_container<HEADER_TYPE, ITEM_TYPE>;             \
    using device_##NAME##_const_container =                           \
        traccc::device_container<const HEADER_TYPE, const ITEM_TYPE>; \
    using NAME##_container_view =                                     \
        traccc::container_view<HEADER_TYPE, ITEM_TYPE>;               \
    using NAME##_container_const_view =                               \
        traccc::container_view<const HEADER_TYPE, const ITEM_TYPE>;   \
    using NAME##_container_data =                                     \
        traccc::container_data<HEADER_TYPE, ITEM_TYPE>;               \
    using NAME##_container_const_data =                               \
        traccc::container_data<const HEADER_TYPE, const ITEM_TYPE>;   \
    using NAME##_container_buffer =                                   \
        traccc::container_buffer<HEADER_TYPE, ITEM_TYPE>
