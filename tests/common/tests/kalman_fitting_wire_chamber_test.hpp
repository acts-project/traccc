/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "kalman_fitting_test.hpp"

// Detray include(s).
#include "detray/detectors/bfield.hpp"
#include "detray/io/frontend/detector_writer.hpp"
#include "detray/test/utils/detectors/build_wire_chamber.hpp"

// System include(s)
#include <array>

namespace traccc {

/// Kalman Fitting Test with Wire Chamber
///
/// Test parameters:
/// (1) name
/// (2) origin
/// (3) origin stddev
/// (4) momentum range
/// (5) eta range
/// (6) phi range
/// (7) particle type
/// (8) number of tracks per event
/// (9) number of events
/// (10) random charge
class KalmanFittingWireChamberTests
    : public KalmanFittingTests,
      public testing::WithParamInterface<std::tuple<
          std::string, std::array<scalar, 3u>, std::array<scalar, 3u>,
          std::array<scalar, 2u>, std::array<scalar, 2u>,
          std::array<scalar, 2u>, detray::pdg_particle<scalar>, unsigned int,
          unsigned int, bool>> {

    public:
    /// Number of layers
    static const inline unsigned int n_wire_layers{20u};

    /// Half z of cylinder
    static const inline scalar half_z{2000.f * detray::unit<scalar>::mm};

    /// B field value and its type
    static constexpr vector3 B{0, 0, 2 * detray::unit<scalar>::T};

    /// Step constraint
    static const inline float step_constraint = 1.f * detray::unit<float>::mm;

    // Set mask tolerance to a large value not to miss the surface during KF
    static const inline scalar mask_tolerance = 75.f * detray::unit<scalar>::um;

    // Grid search window
    static const inline std::array<detray::dindex, 2> search_window{3u, 3u};

    /// Measurement smearing parameters
    static constexpr std::array<scalar, 2u> smearing{
        50.f * detray::unit<scalar>::um, 50.f * detray::unit<scalar>::um};

    /// Standard deviations for seed track parameters
    static constexpr std::array<scalar, e_bound_size> stddevs = {
        0.01f * detray::unit<scalar>::mm,
        0.01f * detray::unit<scalar>::mm,
        0.001f,
        0.001f,
        0.001f / detray::unit<scalar>::GeV,
        0.01f * detray::unit<scalar>::ns};

    void consistency_tests(const track_state_collection_types::host&
                               track_states_per_track) const {

        // The nubmer of track states is supposed be greater than or
        // equal to the number of layers
        ASSERT_GE(track_states_per_track.size(), n_wire_layers);
    }

    protected:
    virtual void SetUp() override {
        vecmem::host_memory_resource host_mr;

        detray::wire_chamber_config wire_chamber_cfg;
        wire_chamber_cfg.n_layers(n_wire_layers);
        wire_chamber_cfg.half_z(half_z);

        //@NOTE: 2 GeV test fails in pull check with the following setup
        // wire_chamber_cfg.mapped_material(detray::beryllium<scalar>());
        // wire_chamber_cfg.m_thickness = 100.f * detray::unit<scalar>::um;

        // Create telescope detector
        auto [det, name_map] = build_wire_chamber(host_mr, wire_chamber_cfg);

        // Write detector file
        auto writer_cfg = detray::io::detector_writer_config{}
                              .format(detray::io::format::json)
                              .replace_files(true)
                              .write_grids(true)
                              .write_material(true)
                              .path(std::get<0>(GetParam()));
        detray::io::write_detector(det, name_map, writer_cfg);
    }
};

}  // namespace traccc
