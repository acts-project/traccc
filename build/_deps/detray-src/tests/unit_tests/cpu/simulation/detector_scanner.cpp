/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "detray/test/validation/detector_scanner.hpp"

#include "detray/definitions/units.hpp"
#include "detray/navigation/detail/trajectories.hpp"
#include "detray/tracks/tracks.hpp"

// Detray test include(s)
#include "detray/test/utils/detectors/build_toy_detector.hpp"
#include "detray/test/utils/simulation/event_generator/track_generators.hpp"
#include "detray/test/utils/types.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

using namespace detray;

using algebra_t = test::algebra;
using vector3 = test::vector3;

constexpr const scalar tol{1e-7f};

/// Brute force test: Intersect toy geometry and compare between ray and helix
/// without B-field
GTEST_TEST(detray_simulation, detector_scanner) {

    // Simulate straight line track
    const vector3 B{0.f * unit<scalar>::T, 0.f * unit<scalar>::T,
                    tol * unit<scalar>::T};

    // Build the geometry
    vecmem::host_memory_resource host_mr;
    auto [toy_det, names] = build_toy_detector(host_mr);

    unsigned int theta_steps{50u};
    unsigned int phi_steps{50u};

    // Record ray tracing
    using detector_t = decltype(toy_det);
    using intersection_trace_t = typename detray::ray_scan<
        algebra_t>::template intersection_trace_type<detector_t>;

    detector_t::geometry_context gctx{};

    std::vector<intersection_trace_t> expected;

    //  Iterate through uniformly distributed momentum directions with ray
    for (const auto test_ray : uniform_track_generator<detail::ray<algebra_t>>(
             phi_steps, theta_steps)) {

        // Record all intersections and objects along the ray
        const auto intersection_record =
            detector_scanner::run<ray_scan>(gctx, toy_det, test_ray);

        expected.push_back(intersection_record);
    }

    // Iterate through uniformly distributed momentum directions with helix
    std::size_t n_tracks{0u};
    for (const auto track :
         uniform_track_generator<free_track_parameters<algebra_t>>(
             phi_steps, theta_steps)) {
        const detail::helix test_helix(track, &B);

        // Record all intersections and objects along the ray
        const auto intersection_trace =
            detector_scanner::run<helix_scan>(gctx, toy_det, test_helix);

        // Should have encountered the same number of tracks (vulnerable to
        // floating point errors)
        EXPECT_EQ(expected[n_tracks].size(), intersection_trace.size())
            << test_helix;

        // Check every single recorded intersection
        for (std::size_t i = 0u;
             i < std::min(expected[n_tracks].size(), intersection_trace.size());
             ++i) {
            if (expected[n_tracks][i].vol_idx !=
                intersection_trace[i].vol_idx) {
                // Intersection record at portal bound might be flipped
                // (the portals overlap completely)
                if (expected[n_tracks][i].vol_idx ==
                        intersection_trace[i + 1u].vol_idx &&
                    expected[n_tracks][i + 1u].vol_idx ==
                        intersection_trace[i].vol_idx) {
                    // Have already checked the next record
                    ++i;
                    continue;
                }
            }
            EXPECT_EQ(expected[n_tracks][i].vol_idx,
                      intersection_trace[i].vol_idx);
        }

        ++n_tracks;
    }
}
