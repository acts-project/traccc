/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL library include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate_collection.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/bfield.hpp"
#include "traccc/utils/detector_type_utils.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <functional>

namespace traccc::sycl {

/// CKF track finding algorithm
class combinatorial_kalman_filter_algorithm
    : public algorithm<edm::track_candidate_collection<default_algebra>::buffer(
          const default_detector::view&,
          const covfie::field<traccc::const_bfield_backend_t<
              default_detector::device::scalar_type>>::view_t&,
          const measurement_collection_types::const_view&,
          const bound_track_parameters_collection_types::const_view&)>,
      public algorithm<edm::track_candidate_collection<default_algebra>::buffer(
          const telescope_detector::view&,
          const covfie::field<traccc::const_bfield_backend_t<
              default_detector::device::scalar_type>>::view_t&,
          const measurement_collection_types::const_view&,
          const bound_track_parameters_collection_types::const_view&)>,
      public messaging {

    public:
    /// Configuration type
    using config_type = finding_config;
    /// Output type
    using output_type =
        edm::track_candidate_collection<default_algebra>::buffer;

    /// Constructor with the algorithm's configuration
    explicit combinatorial_kalman_filter_algorithm(
        const config_type& config, const traccc::memory_resource& mr,
        vecmem::copy& copy, queue_wrapper queue,
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
        const default_detector::view& det,
        const covfie::field<const_bfield_backend_t<
            telescope_detector::device::scalar_type>>::view_t& field,
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
        const telescope_detector::view& det,
        const covfie::field<const_bfield_backend_t<
            telescope_detector::device::scalar_type>>::view_t& field,
        const measurement_collection_types::const_view& measurements,
        const bound_track_parameters_collection_types::const_view& seeds)
        const override;

    private:
    /// Algorithm configuration
    config_type m_config;
    /// Memory resource used by the algorithm
    traccc::memory_resource m_mr;
    /// Copy object used by the algorithm
    std::reference_wrapper<vecmem::copy> m_copy;
    /// Queue wrapper
    mutable queue_wrapper m_queue;
};  // class combinatorial_kalman_filter_algorithm

}  // namespace traccc::sycl
