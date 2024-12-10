/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/geometry/detail/surface_descriptor.hpp"
#include "detray/utils/grid/detail/concepts.hpp"

// System include(s)
#include <concepts>

namespace detray::concepts {

template <class accelerator_t>
concept surface_grid = concepts::grid<accelerator_t>&& std::same_as<
    typename accelerator_t::value_type,
    surface_descriptor<typename accelerator_t::value_type::mask_link,
                       typename accelerator_t::value_type::material_link,
                       typename accelerator_t::value_type::transform_link,
                       typename accelerator_t::value_type::navigation_link>>;

}  // namespace detray::concepts
