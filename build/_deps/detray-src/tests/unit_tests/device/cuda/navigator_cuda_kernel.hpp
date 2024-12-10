/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/units.hpp"
#include "detray/detectors/toy_metadata.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/line_stepper.hpp"
#include "detray/propagator/propagation_config.hpp"

// Detray test include(s)
#include "detray/test/utils/simulation/event_generator/track_generators.hpp"

namespace detray {

using algebra_t = ALGEBRA_PLUGIN<detray::scalar>;
using point3 = dpoint3D<algebra_t>;

// some useful type declarations
using detector_host_t = detector<toy_metadata, host_container_types>;
using detector_device_t = detector<toy_metadata, device_container_types>;

using intersection_t =
    intersection2D<typename detector_device_t::surface_type, algebra_t>;

using navigator_host_t = navigator<detector_host_t>;
using navigator_device_t = navigator<detector_device_t>;
using stepper_t = line_stepper<algebra_t>;

// detector configuration
constexpr std::size_t n_brl_layers{4u};
constexpr std::size_t n_edc_layers{3u};

// geomery navigation configurations
constexpr unsigned int theta_steps{100u};
constexpr unsigned int phi_steps{100u};

constexpr dscalar<algebra_t> pos_diff_tolerance{1e-3f};

// dummy propagator state
template <typename navigation_t>
struct prop_state {
    using context_t = typename navigation_t::detector_type::geometry_context;
    stepper_t::state _stepping;
    navigation_t _navigation;
    context_t _context{};
};

/// test function for navigator with single state
void navigator_test(
    typename detector_host_t::view_type det_data, navigation::config& nav_cfg,
    stepping::config& step_cfg,
    vecmem::data::vector_view<free_track_parameters<algebra_t>>& tracks_data,
    vecmem::data::jagged_vector_view<dindex>& volume_records_data,
    vecmem::data::jagged_vector_view<point3>& position_records_data);

}  // namespace detray
