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
#include "detray/detectors/build_toy_detector.hpp"
#include "detray/io/frontend/detector_writer.hpp"

// System include(s)
#include <array>

namespace traccc {

/// Kalman Finding Test for toy geometry
class KalmanFittingToyDetectorTests : public KalmanFittingTests {

    public:
    /// Number of barrel layers
    static constexpr inline unsigned int n_barrels{4u};

    /// Number of endcap layers
    static constexpr inline unsigned int n_endcaps{7u};

    /// B field value and its type
    static constexpr vector3 B{0, 0, 2 * detray::unit<scalar>::T};

    /// Step constraint
    static const inline scalar step_constraint = 1.f * detray::unit<scalar>::mm;

    /// Measurement smearing parameters
    static constexpr std::array<scalar, 2u> smearing{
        50.f * detray::unit<scalar>::um, 50.f * detray::unit<scalar>::um};

    // Grid search window
    static const inline std::array<detray::dindex, 2> search_window{3u, 3u};

    /// Standard deviations for seed track parameters
    static constexpr std::array<scalar, e_bound_size> stddevs = {
        0.01f * detray::unit<scalar>::mm,
        0.01f * detray::unit<scalar>::mm,
        0.001f,
        0.001f,
        0.01f / detray::unit<scalar>::GeV,
        0.01f * detray::unit<scalar>::ns};

    protected:
    virtual void SetUp() override {
        vecmem::host_memory_resource host_mr;

        detray::toy_det_config toy_cfg{};
        toy_cfg.n_brl_layers(n_barrels).n_edc_layers(n_endcaps).do_check(false);

        // Create the toy geometry
        auto [det, name_map] = detray::build_toy_detector(host_mr, toy_cfg);

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
