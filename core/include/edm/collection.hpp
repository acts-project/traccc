/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/vector.hpp>

#include "definitions/qualifiers.hpp"

namespace traccc {

/// Collection describing objects in a given event
///
/// This is the generic collection of the code, holding a vector of objects.
///
/// It can be instantiated with different vector types, to be able to use
/// the same collection type in both host and device code.
///
/// It also can be instantiated with different edm types represented by
/// item type.
template <typename item_t, template <typename> class vector_t>
class collection {

    public:
    /// @name Type definitions
    /// @{

    /// Vector type used by the collection
    template <typename T>
    using vector_type = vector_t<T>;

    /// The header vector type
    using item_vector = vector_type<item_t>;

    /// @}

    /**
     * @brief The size type of this container, which is the type by which
     * its elements are indexed.
     */
    using size_type = typename item_vector::size_type;

    /**
     * @brief Standard constructor.
     *
     */
    TRACCC_HOST
    collection(item_vector&& iv) : items(iv) {}

    /**
     * @brief Argument-forwarding constructor.
     *
     * This constructor forwards its arguments to item vector.
     * Obviously, this requires the arguments to be quite general, but this
     * constructor can be useful when initializing a collection for a
     * specific vecmem memory resource.
     */
    template <class... args>
    TRACCC_HOST collection(args&&... a) : items(a...) {}

    /**
     * @brief Mutable element accessor.
     */
    TRACCC_HOST_DEVICE
    item_t& operator[](size_type i) { return items.at(i); }

    /**
     * @brief Immutable element accessor.
     */
    TRACCC_HOST_DEVICE
    const item_t& operator[](size_type i) const { return items.at(i); }

    /**
     * @brief Return the size of the collection.
     */
    TRACCC_HOST_DEVICE
    size_type size(void) const { return items.size(); }

    /**
     * @brief Reserve space in both vectors.
     */
    TRACCC_HOST
    void reserve(size_type s) { items.reserve(s); }

    /**
     * @brief Push an element into the collection.
     */
    template <typename i_prime>
    TRACCC_HOST void push_back(i_prime&& new_item) {
        items.push_back(std::forward<item_t>(new_item));
    }

    TRACCC_HOST_DEVICE
    auto begin() { return items.begin(); }

    TRACCC_HOST_DEVICE
    auto end() { return items.end(); }

    /// All objects in the event
    item_vector items;
};

/// Convenience declaration for the collection type to use in host code
template <typename item_t>
using host_collection = collection<item_t, vecmem::vector>;

/// Convenience declaration for the collection type to use in device code
template <typename item_t>
using device_collection = collection<item_t, vecmem::device_vector>;

/// @name Types used to send data back and forth between host and device code
/// @{

/// Structure holding (some of the) data about the collection in host code
template <typename item_t>
struct collection_data {
    vecmem::data::vector_view<item_t> items;
};

/// Structure holding (all of the) data about the collection in host code
template <typename item_t>
struct collection_buffer {
    vecmem::data::vector_buffer<item_t> items;
};

/// Structure used to send the data about the collection to device code
///
/// This is the type that can be passed to device code as-is. But since in
/// host code one needs to manage the data describing a
/// @c traccc::collection either using @c traccc::collection_data or
/// @c traccc::collection_buffer, it needs to have constructors from
/// both of those types.
///
/// In fact it needs to be created from one of those types, as such an
/// object can only function if an instance of one of those types exists
/// alongside it as well.
///
template <typename item_t>
struct collection_view {
    /// Constructor from a @c collection_data object
    collection_view(const collection_data<item_t>& data) : items(data.items) {}

    /// Constructor from a @c collection_buffer object
    collection_view(const collection_buffer<item_t>& buffer)
        : items(buffer.items) {}

    /// View of the data describing the items
    vecmem::data::vector_view<item_t> items;
};

/// Helper function for making a "simple" object out of the collection
template <typename item_t>
inline collection_data<item_t> get_data(host_collection<item_t>& cc) {
    return {vecmem::get_data(cc.items)};
}

}  // namespace traccc
