/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <covfie/core/field.hpp>
#include <traccc/fitting/kalman_filter/kalman_fitter.hpp>
#include <traccc/utils/bfield.hpp>

#include "traccc/utils/propagation.hpp"

namespace traccc {
/// Helper type constructors to facilitate the consistent instantiation of
/// tools such as steppers and fitters for specific detectors.
template <typename detector_t,
          unsigned int cache_size = detray::navigation::default_cache_size>
using navigator_for_t = detray::navigator<const detector_t, cache_size>;

template <typename detector_t>
using stepper_for_t = ::detray::rk_stepper<
    typename ::covfie::field<
        const_bfield_backend_t<typename detector_t::scalar_type>>::view_t,
    typename detector_t::algebra_type,
    ::detray::constrained_step<typename detector_t::scalar_type>>;

template <typename detector_t>
using fitter_for_t =
    kalman_fitter<stepper_for_t<detector_t>, navigator_for_t<detector_t>>;
}  // namespace traccc
