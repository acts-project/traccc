/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_state.hpp"

namespace traccc {

/// Type unrolling functor to update the fitting quality
template <typename algebra_t>
struct statistics_updater {

    using scalar_type = typename algebra_t::scalar_type;

    /// Update track fitting qualities (NDoF and Chi2)
    ///
    /// @param mask_group mask group that contains the mask of the current
    /// surface
    /// @param index mask index of the current surface
    /// @param fit_info fitting information such as NDoF or Chi2
    /// @param trk_state track state of the current surface
    template <typename mask_group_t, typename index_t>
    TRACCC_HOST_DEVICE inline void operator()(
        const mask_group_t& /*mask_group*/, const index_t& /*index*/,
        fitter_info<algebra_t>& fit_info,
        const track_state<algebra_t>& trk_state) {

        if (!trk_state.is_hole) {

            // Measurement dimension
            constexpr const unsigned int D =
                mask_group_t::value_type::shape::meas_dim;

            // NDoF = NDoF + number of coordinates per measurement
            fit_info.ndf += static_cast<scalar_type>(D);

            // total_chi2 = total_chi2 + chi2
            fit_info.chi2 += trk_state.smoothed_chi2();
        }
    }
};

}  // namespace traccc