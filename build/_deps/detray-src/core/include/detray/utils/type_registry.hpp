/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/indexing.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/utils/type_list.hpp"

// System include(s)
#include <type_traits>
#include <utility>

namespace detray {

/// @brief match types with indices and vice versa.
///
/// @tparam IDs enum that references the types (not used in base class)
/// @tparam registered_types the types that can be mapped to indices
template <class ID, typename... registered_types>
class type_registry {
    public:
    // Make the type IDs accessible
    using id = ID;

    /// Conventions for some basic info
    enum : std::size_t {
        n_types = sizeof...(registered_types),
        e_any = sizeof...(registered_types),
        e_unknown = sizeof...(registered_types) + 1,
    };

    /// Get the index for a type. Needs to be unrolled in case of thrust tuple.
    template <typename object_t>
    DETRAY_HOST_DEVICE static constexpr ID get_id() {
        return unroll_ids<std::decay_t<object_t>, registered_types...>();
    }

    /// Get the index for a type. Use template parameter deduction.
    template <typename object_t>
    DETRAY_HOST_DEVICE static constexpr ID get_id(const object_t& /*obj*/) {
        return get_id<object_t>();
    }

    /// Checks whether a given types is known in the registry.
    /// Use template parameter deduction.
    template <typename object_t>
    DETRAY_HOST_DEVICE static constexpr bool is_defined(
        const object_t& /*obj*/) {
        return (get_id<object_t>() != static_cast<ID>(e_unknown));
    }

    /// Checks whether a given types is known in the registry.
    template <typename object_t>
    DETRAY_HOST_DEVICE static constexpr bool is_defined() {
        return (get_id<object_t>() != static_cast<ID>(e_unknown));
    }

    /// Checks whether a given index can be mapped to a type.
    DETRAY_HOST_DEVICE static constexpr bool is_valid(
        const std::size_t type_id) {
        return type_id < n_types;
    }

    /// Convert index to ID and do some (limited) checking.
    ///
    /// @tparam ref_idx matches to index arg to perform static checks
    /// @param index argument to be converted to valid id type
    ///
    /// @return the matching ID type.
    template <std::size_t ref_idx = 0>
    DETRAY_HOST_DEVICE static constexpr ID to_id(const std::size_t index) {
        if (ref_idx == index) {
            // Produce a more helpful error than the usual tuple index error
            static_assert(
                is_valid(ref_idx),
                "Index out of range: Please make sure that indices and type "
                "enums match the number of types in container.");
            return static_cast<ID>(ref_idx);
        }
        if constexpr (ref_idx < sizeof...(registered_types) - 1) {
            return to_id<ref_idx + 1>(index);
        }
        // This produces a compiler error when used in type unrolling code
        return static_cast<ID>(sizeof...(registered_types));
    }

    /// Convert index to ID and do some (limited) checking.
    ///
    /// @tparam ref_idx matches to index arg to perform static checks
    /// @param index argument to be converted to valid id type
    ///
    /// @return the matching ID type.
    template <std::size_t ref_idx = 0>
    DETRAY_HOST_DEVICE static constexpr std::size_t to_index(const ID id) {
        if (to_id(ref_idx) == id) {
            // Produce a more helpful error than the usual tuple index error
            static_assert(
                is_valid(ref_idx),
                "Index out of range: Please make sure that indices and type "
                "enums match the number of types in container.");
            return ref_idx;
        }
        if constexpr (ref_idx < sizeof...(registered_types) - 1) {
            return to_index<ref_idx + 1>(id);
        }
        // This produces a compiler error when used in type unrolling code
        return sizeof...(registered_types);
    }

    /// Extract an index and check it.
    template <typename object_t>
    struct get_index {
        static constexpr ID value = get_id<object_t>();
        DETRAY_HOST_DEVICE
        constexpr bool operator()() const noexcept { return is_valid(value); }
    };

    /// Return a type for an index. If the index cannot be mapped, there will be
    /// a compiler error.
    template <ID type_id>
    struct get_type {
        using type = types::at<types::list<registered_types...>,
                               static_cast<int>(to_index(type_id))>;
    };

    private:
    /// dummy type
    struct empty_type {};

    /// Gets the position of a type in a parameter pack, without using tuples.
    template <typename object_t, typename first_t = empty_type,
              typename... remaining_types>
    DETRAY_HOST_DEVICE static constexpr ID unroll_ids() {
        if constexpr (!std::is_same_v<first_t, empty_type> &&
                      !std::is_same_v<object_t, first_t>) {
            return unroll_ids<object_t, remaining_types...>();
        }
        if constexpr (std::is_same_v<object_t, first_t>) {
            return static_cast<ID>(n_types - sizeof...(remaining_types) - 1);
        }
        return static_cast<ID>(e_unknown);
    }
};

}  // namespace detray
