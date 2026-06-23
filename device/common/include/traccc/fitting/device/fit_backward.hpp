/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/global_index.hpp"
#include "traccc/fitting/device/fit_payload.hpp"

namespace traccc::device {

/// Function performing a backward fit iteration
template <typename fitter_t>
TRACCC_HOST_DEVICE inline void fit_backward(
    const global_index_t globalIndex, const typename fitter_t::config_type& cfg,
    const fit_payload& payload,
    const fit_tpayload<typename fitter_t::detector_type::const_view_type,
                       typename fitter_t::bfield_type,
                       typename fitter_t::surface_type>& tpayload);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/fitting/device/impl/fit_backward.ipp"
