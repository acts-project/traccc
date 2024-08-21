/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_parameters.hpp"

// detray include(s).
#include "detray/definitions/pdg_particle.hpp"

// System include(s).
#include <stdexcept>

namespace traccc {

namespace detail {

template <typename scalar_t>
TRACCC_HOST_DEVICE inline detray::pdg_particle<scalar_t>
particle_from_pdg_number(const int pdg_num) {

    switch (pdg_num) {
        case 11:
            return detray::electron<scalar_t>();
        case -11:
            return detray::positron<scalar_t>();
        case 13:
            return detray::muon<scalar_t>();
        case -13:
            return detray::antimuon<scalar_t>();
        case 211:
            return detray::pion_plus<scalar_t>();
        case -211:
            return detray::pion_minus<scalar_t>();
    }

    return detray::muon<scalar_t>();
}

// Apply the charge operator to return the antimatter
template <typename scalar_t>
TRACCC_HOST_DEVICE inline detray::pdg_particle<scalar_t> charge_conjugation(
    const detray::pdg_particle<scalar_t>& ptc) {

    const auto pdg_num = ptc.pdg_num();

    switch (pdg_num) {
        case 11:
            return detray::positron<scalar_t>();
        case -11:
            return detray::electron<scalar_t>();
        case 13:
            return detray::antimuon<scalar_t>();
        case -13:
            return detray::muon<scalar_t>();
        case 211:
            return detray::pion_minus<scalar_t>();
        case -211:
            return detray::pion_plus<scalar_t>();
    }

    return detray::muon<scalar_t>();
}

// Return the consistent particle type based on the particle hypothesis and the
// charge of the track parameters
template <typename scalar_t>
TRACCC_HOST_DEVICE inline detray::pdg_particle<scalar_t>
correct_particle_hypothesis(
    const detray::pdg_particle<scalar_t>& ptc_hypothesis,
    const bound_track_parameters& params) {

    if (ptc_hypothesis.charge() * params.qop() > 0.f) {
        return ptc_hypothesis;
    } else {
        return charge_conjugation(ptc_hypothesis);
    }
}

template <typename scalar_t>
TRACCC_HOST_DEVICE inline detray::pdg_particle<scalar_t>
correct_particle_hypothesis(
    const detray::pdg_particle<scalar_t>& ptc_hypothesis,
    const free_track_parameters& params) {

    if (ptc_hypothesis.charge() * params.qop() > 0.f) {
        return ptc_hypothesis;
    } else {
        return charge_conjugation(ptc_hypothesis);
    }
}

}  // namespace detail

}  // namespace traccc