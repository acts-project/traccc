/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/finding/actors/ckf_aborter.hpp"
#include "traccc/finding/actors/interaction_register.hpp"

// Detray include(s).
#include <detray/navigation/navigator.hpp>
#include <detray/propagator/actor_chain.hpp>
#include <detray/propagator/actors/aborters.hpp>
#include <detray/propagator/actors/parameter_resetter.hpp>
#include <detray/propagator/actors/parameter_transporter.hpp>
#include <detray/propagator/actors/pointwise_material_interactor.hpp>
#include <detray/propagator/constrained_step.hpp>
#include <detray/propagator/propagator.hpp>
#include <detray/propagator/rk_stepper.hpp>

// System include(s).
#include <type_traits>

namespace traccc::details {

/// Stepper used in the Combinatorial Kalman Filter (CKF)
///
/// @tparam bfield_t The type of magnetic field to use
///
template <typename bfield_t>
using ckf_stepper_t = detray::rk_stepper<
    bfield_t, traccc::default_algebra, detray::constrained_step<traccc::scalar>,
    detray::stepper_rk_policy<traccc::scalar>, detray::stepping::void_inspector,
    static_cast<std::uint32_t>(
        detray::rk_stepper_flags::e_allow_covariance_transport)>;

/// Interactor used in the Combinatorial Kalman Filter (CKF)
using ckf_interactor_t =
    detray::pointwise_material_interactor<traccc::default_algebra>;

/// Actor chain used in the Combinatorial Kalman Filter (CKF)
using ckf_actor_chain_t =
    detray::actor_chain<detray::pathlimit_aborter<traccc::scalar>,
                        detray::parameter_transporter<traccc::default_algebra>,
                        interaction_register<ckf_interactor_t>,
                        ckf_interactor_t,
                        detray::parameter_resetter<traccc::default_algebra>,
                        detray::momentum_aborter<traccc::scalar>, ckf_aborter>;

/// Propagator type used in the Combinatorial Kalman Filter (CKF)
///
/// @tparam detector_t The detector type to use
///
template <typename detector_t, typename bfield_t>
using ckf_propagator_t =
    detray::propagator<ckf_stepper_t<bfield_t>,
                       detray::navigator<std::add_const_t<detector_t>>,
                       ckf_actor_chain_t>;

}  // namespace traccc::details
