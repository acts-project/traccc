/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/definitions/units.hpp"
#include "detray/navigation/detail/ray.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/line_stepper.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/test/utils/detectors/build_toy_detector.hpp"
#include "detray/test/utils/inspectors.hpp"
#include "detray/test/utils/simulation/event_generator/track_generators.hpp"
#include "detray/test/validation/detector_scanner.hpp"
#include "detray/tracks/tracks.hpp"

// Example linear algebra plugin: std::array
#include "detray/tutorial/types.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s)
#include <sstream>

/// Run the navigation through the toy detector including a detailed record
/// of the encountered surfaces (using the navigation inspectors)
int main() {
    // Toy detector
    using toy_detector_t = detray::detector<detray::toy_metadata>;

    /// Type that holds the intersection information
    using intersection_t =
        detray::intersection2D<typename toy_detector_t::surface_type,
                               detray::tutorial::algebra_t, true>;

    /// Inspector that records all encountered surfaces
    using object_tracer_t = detray::navigation::object_tracer<
        intersection_t, detray::dvector,
        detray::navigation::status::e_on_module,
        detray::navigation::status::e_on_portal>;

    /// Inspector that prints the navigator state from within the navigator's
    /// method calls (cannot be done with an actor)
    using nav_print_inspector_t = detray::navigation::print_inspector;

    /// Aggregation of multiple inspectors
    using inspector_t =
        detray::aggregate_inspector<object_tracer_t, nav_print_inspector_t>;
    /// How many intersections to cache during navigation
    constexpr std::size_t cache_size{detray::navigation::default_cache_size};

    // Navigation with inspection
    using navigator_t = detray::navigator<toy_detector_t, cache_size,
                                          inspector_t, intersection_t>;
    // Line stepper
    using stepper_t = detray::line_stepper<detray::tutorial::algebra_t>;
    // Propagator with empty actor chain
    using propagator_t =
        detray::propagator<stepper_t, navigator_t, detray::actor_chain<>>;

    vecmem::host_memory_resource host_mr;

    const auto [det, names] = detray::build_toy_detector(host_mr);

    typename toy_detector_t::geometry_context gctx{};

    // Build the propagator
    detray::propagation::config prop_cfg{};
    propagator_t prop{prop_cfg};

    // Track generation config
    // Trivial example: Single track escapes through beampipe
    using ray_type = detray::detail::ray<detray::tutorial::algebra_t>;
    constexpr std::size_t theta_steps{1};
    constexpr std::size_t phi_steps{1};

    // Iterate through uniformly distributed momentum directions
    for (const auto ray :
         detray::uniform_track_generator<ray_type>(phi_steps, theta_steps)) {

        // Shoot ray through the detector and record all surface intersections
        const auto intersection_trace =
            detray::detector_scanner::run<detray::ray_scan>(gctx, det, ray);

        // Now follow that ray with the same track and check, if we find
        // the same volumes and distances along the way
        detray::free_track_parameters<detray::tutorial::algebra_t> track(
            ray.pos(), 0.f, ray.dir(), -1.f);
        propagator_t::state propagation(track, det, prop_cfg.context);

        // Run the actual propagation
        prop.propagate(propagation);

        // Retrieve navigation information
        auto &inspector = propagation._navigation.inspector();
        auto &obj_tracer = inspector.template get<object_tracer_t>();
        auto &debug_printer = inspector.template get<nav_print_inspector_t>();

        std::cout << debug_printer.to_string();

        // Compare ray trace to object tracer
        std::stringstream debug_stream;
        for (std::size_t intr_idx = 0; intr_idx < intersection_trace.size();
             ++intr_idx) {
            debug_stream << "-------Intersection trace\n"
                         << "ray gun: "
                         << "found in vol: "
                         << intersection_trace[intr_idx].vol_idx << ",\n\t"
                         << intersection_trace[intr_idx].intersection;
            debug_stream << "\nnavig.:\t" << obj_tracer[intr_idx].intersection;
        }
        std::cout << debug_stream.str() << std::endl;

        // Compare intersection records
        //
        // [...]
        //
    }
}
