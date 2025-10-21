/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/geometry/host_detector.hpp"
#include "traccc/utils/logging.hpp"
#include "traccc/utils/particle.hpp"
#include "traccc/utils/seed_generator.hpp"

// Detray include(s)
#include <detray/propagator/propagation_config.hpp>

// System include(s)
#include <memory>
#include <string>

namespace traccc {

/// Run a detray propagation with and without Kalman filter and compare the
/// surfaces that were encountered against a reference of truth tracks read
/// from file
///
/// @param det the detector
/// @param names the detector and volume names
/// @param prop_cfg propagation configuration, e.g. mask tolerances
/// @param input_dir the truth data path
/// @param n_events how many events to check
/// @param ilogger logging service
/// @param do_multiple_scattering adjust covariances due to multiple scattering
/// during test propagation
/// @param do_energy_loss adjust covariances and track momentum during test
/// propagation
/// @param use_acts_geoid whether to expect ACTS geometry ids in the truth data
/// @param ptc_type the particle hypothesis to use
/// @param stddevs the initial track parameters uncertainty
/// @param B constant magnetic field vector
/// @param min_pT minimum transverse momentum
/// @param min_rad minimum radial distance of the particle vertex position
///
/// @returns whether the validation was successful

bool kalman_filter_comparison(
    const traccc::host_detector* host_det,
    const traccc::default_detector::host::name_map& names,
    const traccc::magnetic_field& bfield,
    const detray::propagation::config& prop_cfg, const std::string& input_dir,
    const unsigned int n_events, std::unique_ptr<const traccc::Logger> ilogger,
    const bool do_multiple_scattering = true, const bool do_energy_loss = true,
    const bool use_acts_geoid = false,
    const traccc::pdg_particle<traccc::scalar> ptc_type =
        traccc::muon<traccc::scalar>(),
    const traccc::seed_generator<traccc::default_detector::host>::config&
        smearing_cfg = {},
    const traccc::scalar min_pT = 0.9f * traccc::unit<traccc::scalar>::GeV,
    const traccc::scalar max_rad = 10.f * traccc::unit<traccc::scalar>::mm);

}  // namespace traccc
