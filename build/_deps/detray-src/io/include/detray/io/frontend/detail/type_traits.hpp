/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/materials/detail/concepts.hpp"
#include "detray/materials/material_rod.hpp"
#include "detray/materials/material_slab.hpp"
#include "detray/navigation/accelerators/concepts.hpp"
#include "detray/utils/grid/detail/concepts.hpp"
#include "detray/utils/type_registry.hpp"
#include "detray/utils/type_traits.hpp"

// System include(s)
#include <type_traits>

namespace detray {

namespace detail {

/// Check for grid types in a data store
/// @{
template <typename>
struct contains_grids;

template <class ID, typename... Ts>
struct contains_grids<type_registry<ID, Ts...>> {
    static constexpr bool value{(concepts::grid<Ts> || ...)};
};

template <typename T>
inline constexpr bool contains_grids_v = contains_grids<T>::value;
/// @}

/// Check for surface grid types in a data store
/// @{
template <typename>
struct contains_surface_grids;

template <class ID, typename... Ts>
struct contains_surface_grids<type_registry<ID, Ts...>> {
    static constexpr bool value{(concepts::surface_grid<Ts> || ...)};
};

template <typename T>
inline constexpr bool contains_surface_grids_v =
    contains_surface_grids<T>::value;
/// @}

/// Check for the various types of material
/// @{

/// Contains slabs
/// @{
template <typename>
struct contains_material_slabs {};

template <class ID, typename... Ts>
struct contains_material_slabs<type_registry<ID, Ts...>> {
    static constexpr bool value{(concepts::material_slab<Ts> || ...)};
};

template <typename T>
inline constexpr bool contains_material_slabs_v =
    contains_material_slabs<T>::value;
/// @}

/// Contains rods
/// @{
template <typename>
struct contains_material_rods {};

template <class ID, typename... Ts>
struct contains_material_rods<type_registry<ID, Ts...>> {
    static constexpr bool value{(concepts::material_rod<Ts> || ...)};
};

template <typename T>
inline constexpr bool contains_material_rods_v =
    contains_material_rods<T>::value;
/// @}

/// Contains homogeneous material
/// @{
template <typename>
struct contains_homogeneous_material {};

template <class ID, typename... Ts>
struct contains_homogeneous_material<type_registry<ID, Ts...>> {
    static constexpr bool value{(concepts::homogeneous_material<Ts> || ...)};
};

template <typename T>
inline constexpr bool contains_homogeneous_material_v =
    contains_homogeneous_material<T>::value;
/// @}

/// Contains material maps
/// @{
template <typename>
struct contains_material_maps {};

template <class ID, typename... Ts>
struct contains_material_maps<type_registry<ID, Ts...>> {
    static constexpr bool value{(concepts::material_map<Ts> || ...)};
};

template <typename T>
inline constexpr bool contains_material_maps_v =
    contains_material_maps<T>::value;
/// @}

}  // namespace detail

namespace concepts {

/// Check for the the presence of any type of grids in a detector definition
template <class detector_t>
concept has_grids = detail::contains_grids_v<typename detector_t::accel> ||
                    detail::contains_grids_v<typename detector_t::materials>;

/// Check for the the presence of surface grids in a detector definition
template <class detector_t>
concept has_surface_grids =
    detail::contains_surface_grids_v<typename detector_t::accel>;

/// Check for the the presence of material slabs in a detector definition
template <class detector_t>
concept has_material_slabs =
    detail::contains_material_slabs_v<typename detector_t::materials>;

/// Check for the the presence of material rods in a detector definition
template <class detector_t>
concept has_material_rods =
    detail::contains_material_rods_v<typename detector_t::materials>;

/// Check for the the presence of homogeneous material types in a detector
/// definition
template <class detector_t>
concept has_homogeneous_material =
    detail::contains_homogeneous_material_v<typename detector_t::materials>;

/// Check for the the presence of material maps in a detector definition
template <class detector_t>
concept has_material_maps =
    detail::contains_material_maps_v<typename detector_t::materials>;

}  // namespace concepts

}  // namespace detray
