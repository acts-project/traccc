/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/cuda/utils/algorithm_base.hpp"
#include "traccc/cuda/utils/stream_wrapper.hpp"

// Project include(s).
#include "traccc/fitting/device/kalman_fitting_algorithm.hpp"

namespace traccc::cuda {

/// Kalman filter based track fitting algorithm using CUDA
class kalman_fitting_algorithm : public device::kalman_fitting_algorithm,
                                 public cuda::algorithm_base {

    public:
    /// Constructor with the algorithm's configuration
    ///
    /// @param config The configuration object
    /// @param mr     The memory resource(s) used by the algorithm
    /// @param copy   The copy object used by the algorithm
    /// @param str    The CUDA stream used by the algorithm
    /// @param logger The logger used by the algorithm
    ///
    kalman_fitting_algorithm(
        const config_type& config, const traccc::memory_resource& mr,
        const vecmem::copy& copy, const stream_wrapper& str,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    private:
    /// @name Function(s) implemented from @c device::kalman_fitting_algorithm
    /// @{

    /// Prepare a buffer with the index order with which to fit the tracks
    ///
    /// @param[in] tracks The tracks to be fitted
    /// @param[out] track_sort_keys Buffer storing temporary sorting keys
    /// @param[out] track_indices The buffer to write the fitting order into
    ///
    void prepare_track_fit_order(
        const edm::track_collection<default_algebra>::const_view& tracks,
        vecmem::data::vector_view<device::sort_key>& track_sort_keys,
        vecmem::data::vector_view<unsigned int>& track_indices) const override;

    /// Kernel to prepare the fitting payloads
    ///
    /// @param[in] track_indices The fitting order of the tracks
    /// @param[in] input_tracks The input tracks
    /// @param[out] output_tracks The output (fitted) tracks
    /// @param[out] track_liveness The buffer to write the track liveness into
    ///
    void fit_prelude_kernel(
        const vecmem::data::vector_view<const unsigned int>& track_indices,
        const edm::track_container<default_algebra>::const_view& input_tracks,
        edm::track_container<default_algebra>::view output_tracks,
        vecmem::data::vector_view<unsigned int>& track_liveness) const override;

    /// Function preparing the fitting payload
    ///
    /// @param det             The detector buffer to prepare the payload for
    /// @param field           The magnetic field to prepare the payload for
    /// @param n_surfaces      The number of surfaces for each track to be
    ///                        fitted
    /// @param track_indices   The fitting order of the tracks
    /// @param track_liveness  The buffer to write the track liveness into
    /// @param tracks          The tracks to be fitted
    ///
    /// @return The prepared payload for the fitting kernel(s)
    ///
    std::unique_ptr<fit_payload_base> prepare_fit_payload(
        const detector_buffer& det, const magnetic_field& field,
        const std::vector<unsigned int>& n_surfaces,
        const vecmem::data::vector_view<const unsigned int>& track_indices,
        vecmem::data::vector_view<unsigned int>& track_liveness,
        edm::track_container<default_algebra>::view tracks) const override;

    /// Function launching the "forward fitting" kernel(s)
    ///
    /// @param config The fitting configuration
    /// @param payload The payload for the fitting kernel(s)
    ///
    void fit_forward_kernel(const fitting_config& config,
                            const fit_payload_base& payload) const override;

    /// Function launching the "backward fitting" kernel(s)
    ///
    /// @param config The fitting configuration
    /// @param payload The payload for the fitting kernel(s)
    ///
    void fit_backward_kernel(const fitting_config& config,
                             const fit_payload_base& payload) const override;

    /// @}

};  // class kalman_fitting_algorithm

}  // namespace traccc::cuda
