/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/messaging.hpp"

// Detray include(s).
#include <detray/detectors/bfield.hpp>

namespace traccc::host {

/// CKF track finding algorithm
///
/// This is the main host-based track finding algorithm of the project. More
/// documentation to be written later...
///
class combinatorial_kalman_filter_algorithm
    : public algorithm<track_candidate_container_types::host(
          const default_detector::host&,
          const detray::bfield::const_field_t<
              default_detector::host::scalar_type>::view_t&,
          const measurement_collection_types::const_view&,
          const bound_track_parameters_collection_types::const_view&)>,
      public algorithm<track_candidate_container_types::host(
          const telescope_detector::host&,
          const detray::bfield::const_field_t<
              telescope_detector::host::scalar_type>::view_t&,
          const measurement_collection_types::const_view&,
          const bound_track_parameters_collection_types::const_view&)>,
      public messaging {

    public:
    /// Configuration type
    using config_type = finding_config;
    /// Output type
    using output_type = track_candidate_container_types::host;

    /// Constructor with the algorithm's configuration
    explicit combinatorial_kalman_filter_algorithm(
        const config_type& config,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    /// Execute the algorithm
    ///
    /// @param det          The (default) detector object
    /// @param field        The (constant) magnetic field object
    /// @param measurements All measurements in an event
    /// @param seeds        All seeds in an event to start the track finding
    ///                     with
    ///
    /// @return A container of the found track candidates
    ///
    output_type operator()(
        const default_detector::host& det,
        const detray::bfield::const_field_t<
            default_detector::host::scalar_type>::view_t& field,
        const measurement_collection_types::const_view& measurements,
        const bound_track_parameters_collection_types::const_view& seeds)
        const override;

    /// Execute the algorithm
    ///
    /// @param det          The (telescope) detector object
    /// @param field        The (constant) magnetic field object
    /// @param measurements All measurements in an event
    /// @param seeds        All seeds in an event to start the track finding
    ///                     with
    ///
    /// @return A container of the found track candidates
    ///
    output_type operator()(
        const telescope_detector::host& det,
        const detray::bfield::const_field_t<
            telescope_detector::host::scalar_type>::view_t& field,
        const measurement_collection_types::const_view& measurements,
        const bound_track_parameters_collection_types::const_view& seeds)
        const override;

    private:
    /// Algorithm configuration
    config_type m_config;
};  // class combinatorial_kalman_filter_algorithm

}  // namespace traccc::host
