/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/sycl/seeding/seeding_algorithm.hpp"

// Project include(s).
#include "traccc/seeding/detail/seeding_config.hpp"

// System include(s).
#include <cmath>

namespace {

/// Helper function that would produce a default seed-finder configuration
traccc::seedfinder_config default_seedfinder_config() {

    traccc::seedfinder_config config;
    config.highland = 13.6 * std::sqrt(config.radLengthPerSeed) *
                      (1 + 0.038 * std::log(config.radLengthPerSeed));
    float maxScatteringAngle = config.highland / config.minPt;
    config.maxScatteringAngle2 = maxScatteringAngle * maxScatteringAngle;
    // helix radius in homogeneous magnetic field. Units are Kilotesla, MeV
    // and millimeter
    // TODO: change using ACTS units
    config.pTPerHelixRadius = 300. * config.bFieldInZ;
    config.minHelixDiameter2 =
        std::pow(config.minPt * 2 / config.pTPerHelixRadius, 2);
    config.pT2perRadius =
        std::pow(config.highland / config.pTPerHelixRadius, 2);
    return config;
}

/// Helper function that would produce a default spacepoint grid configuration
traccc::spacepoint_grid_config default_spacepoint_grid_config() {

    traccc::seedfinder_config config = default_seedfinder_config();
    traccc::spacepoint_grid_config grid_config;
    grid_config.bFieldInZ = config.bFieldInZ;
    grid_config.minPt = config.minPt;
    grid_config.rMax = config.rMax;
    grid_config.zMax = config.zMax;
    grid_config.zMin = config.zMin;
    grid_config.deltaRMax = config.deltaRMax;
    grid_config.cotThetaMax = config.cotThetaMax;
    return grid_config;
}

}  // namespace

namespace traccc::sycl {

seeding_algorithm::seeding_algorithm(const traccc::memory_resource& mr,
                                     const queue_wrapper& queue)
    : m_spacepoint_binning(default_seedfinder_config(),
                           default_spacepoint_grid_config(), mr, queue),
      m_seed_finding(default_seedfinder_config(), mr, queue) {}

vecmem::data::vector_buffer<seed> seeding_algorithm::operator()(
    const spacepoint_container_types::const_view& spacepoints_view) const {

    return m_seed_finding(spacepoints_view,
                          m_spacepoint_binning(spacepoints_view));
}

vecmem::data::vector_buffer<seed> seeding_algorithm::operator()(
    const spacepoint_container_types::buffer& spacepoints_buffer) const {

    return m_seed_finding(spacepoints_buffer,
                          m_spacepoint_binning(spacepoints_buffer));
}

}  // namespace traccc::sycl
