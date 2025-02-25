/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/fitting_config.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/messaging.hpp"

// Detray include(s).
#include <detray/detectors/bfield.hpp>

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <functional>

namespace traccc::host {

/// Kalman filter based track fitting algorithm
class kalman_fitting_algorithm
    : public algorithm<track_state_container_types::host(
          const default_detector::host&,
          const detray::bfield::const_field_t<
              default_detector::host::scalar_type>::view_t&,
          const track_candidate_container_types::const_view&)>,
      public algorithm<track_state_container_types::host(
          const telescope_detector::host&,
          const detray::bfield::const_field_t<
              telescope_detector::host::scalar_type>::view_t&,
          const track_candidate_container_types::const_view&)>,
      public messaging {

    public:
    /// Configuration type
    using config_type = fitting_config;
    /// Output type
    using output_type = track_state_container_types::host;

    /// Constructor with the algorithm's configuration
    ///
    /// @param config The configuration object
    ///
    explicit kalman_fitting_algorithm(
        const config_type& config, vecmem::memory_resource& mr,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    /// Execute the algorithm
    ///
    /// @param det             The (default) detector object
    /// @param field           The (constant) magnetic field object
    /// @param track_candidates All track candidates to fit
    ///
    /// @return A container of the fitted track states
    ///
    output_type operator()(
        const default_detector::host& det,
        const detray::bfield::const_field_t<
            default_detector::host::scalar_type>::view_t& field,
        const track_candidate_container_types::const_view& track_candidates)
        const override;

    /// Execute the algorithm
    ///
    /// @param det             The (telescope) detector object
    /// @param field           The (constant) magnetic field object
    /// @param track_candidates All track candidates to fit
    ///
    /// @return A container of the fitted track states
    ///
    output_type operator()(
        const telescope_detector::host& det,
        const detray::bfield::const_field_t<
            telescope_detector::host::scalar_type>::view_t& field,
        const track_candidate_container_types::const_view& track_candidates)
        const override;

    private:
    /// Algorithm configuration
    config_type m_config;
    /// Memory resource to use in the algorithm
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};  // class kalman_fitting_algorithm

}  // namespace traccc::host
