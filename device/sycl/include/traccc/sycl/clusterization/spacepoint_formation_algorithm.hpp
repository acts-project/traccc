/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL library include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"
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
class spacepoint_formation_algorithm
    : public algorithm<spacepoint_collection_types::buffer(
          const measurement_collection_types::const_view&,
          const silicon_detector_description::const_view&)> {

    public:
    /// Constructor for spacepoint_formation
    ///
    /// @param mr is the memory resource
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param queue is a wrapper for the for the sycl queue for kernel
    ///              invocation
    ///
    spacepoint_formation_algorithm(const traccc::memory_resource& mr,
                                   vecmem::copy& copy, queue_wrapper& queue);

    /// Callable operator for the space point formation, based on one single
    /// module
    ///
    /// @param measurements All reconstructed measurements in an event
    /// @param det_descr    The detector description
    /// @return A spacepoint container, with one spacepoint for every
    ///         measurement
    ///
    output_type operator()(
        const measurement_collection_types::const_view& measurements_view,
        const silicon_detector_description::const_view& det_descr)
        const override;

    private:
    /// Memory resource(s) to use in the algorithm
    traccc::memory_resource m_mr;
    /// The copy object to use
    std::reference_wrapper<vecmem::copy> m_copy;
    /// The SYCL queue to use
    std::reference_wrapper<queue_wrapper> m_queue;

    /// The maximum number of threads in a work group
    unsigned int m_max_work_group_size;

};  // class spacepoint_formation_algorithm

}  // namespace traccc::sycl
