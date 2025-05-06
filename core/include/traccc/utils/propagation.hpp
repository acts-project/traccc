/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/track_parameters.hpp"

// Detray include(s).
#include <detray/navigation/direct_navigator.hpp>
#include <detray/navigation/navigator.hpp>
#include <detray/propagator/actors.hpp>
// TODO: Remove when updating to future detray version.
#include <detray/propagator/constrained_step.hpp>
#include <detray/propagator/line_stepper.hpp>
#include <detray/propagator/propagator.hpp>
#include <detray/propagator/rk_stepper.hpp>
