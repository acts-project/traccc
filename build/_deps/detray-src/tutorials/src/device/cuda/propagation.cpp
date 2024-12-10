/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "propagation.hpp"

#include "detray/test/utils/detectors/build_toy_detector.hpp"
#include "detray/test/utils/simulation/event_generator/track_generators.hpp"

// Vecmem include(s)
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// System

/// Prepare the data and move it to device
int main() {
    // VecMem memory resource(s)
    vecmem::cuda::managed_memory_resource mng_mr;

    // Create the host bfield
    auto bfield = detray::bfield::create_inhom_field();

    // Create the toy geometry
    auto [det, names] = detray::build_toy_detector(mng_mr);

    // Create the vector of initial track parameters
    vecmem::vector<detray::free_track_parameters<detray::tutorial::algebra_t>>
        tracks(&mng_mr);

    // Track directions to be generated
    constexpr unsigned int theta_steps{10u};
    constexpr unsigned int phi_steps{10u};
    // Set momentum of tracks
    const detray::scalar p_mag{10.f * detray::unit<detray::scalar>::GeV};

    // Genrate the tracks
    for (auto track : detray::uniform_track_generator<
             detray::free_track_parameters<detray::tutorial::algebra_t>>(
             phi_steps, theta_steps, p_mag)) {
        // Put it into vector of tracks
        tracks.push_back(track);
    }

    // Get data for device
    auto det_data = detray::get_data(det);
    covfie::field<detray::tutorial::bfield::cuda::inhom_bknd_t> device_bfield(
        bfield);
    auto tracks_data = detray::get_data(tracks);

    // Run the propagator test for GPU device
    detray::tutorial::propagation(det_data, device_bfield, tracks_data);
}
