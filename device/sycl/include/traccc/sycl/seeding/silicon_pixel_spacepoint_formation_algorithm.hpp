/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/sycl/utils/queue_wrapper.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <functional>

namespace traccc::sycl {

/// Algorithm forming space points out of measurements
///
/// This algorithm performs the local-to-global transformation of the 2D
/// measurements made on every detector module, into 3D spacepoint coordinates.
///
class silicon_pixel_spacepoint_formation_algorithm
    : public algorithm<spacepoint_collection_types::buffer(
          const default_detector::view&,
          const measurement_collection_types::const_view&)>,
      public algorithm<spacepoint_collection_types::buffer(
          const telescope_detector::view&,
          const measurement_collection_types::const_view&)> {

    public:
    /// Output type
    using output_type = spacepoint_collection_types::buffer;

    /// Constructor for spacepoint_formation
    ///
    /// @param mr is the memory resource
    ///
    silicon_pixel_spacepoint_formation_algorithm(
        const traccc::memory_resource& mr, vecmem::copy& copy,
        queue_wrapper queue);

    /// Construct spacepoints from 2D silicon pixel measurements
    ///
    /// @param det Detector object
    /// @param measurements A collection of measurements
    /// @return A spacepoint buffer, with one spacepoint for every
    ///         silicon pixel measurement
    ///
    output_type operator()(const default_detector::view& det,
                           const measurement_collection_types::const_view&
                               measurements) const override;

    /// Construct spacepoints from 2D silicon pixel measurements
    ///
    /// @param det Detector object
    /// @param measurements A collection of measurements
    /// @return A spacepoint buffer, with one spacepoint for every
    ///         silicon pixel measurement
    ///
    output_type operator()(const telescope_detector::view& det,
                           const measurement_collection_types::const_view&
                               measurements) const override;

    private:
    /// Memory resource used by the algorithm
    traccc::memory_resource m_mr;
    /// The copy object to use
    std::reference_wrapper<vecmem::copy> m_copy;
    /// SYCL queue object
    mutable queue_wrapper m_queue;
};

}  // namespace traccc::sycl
