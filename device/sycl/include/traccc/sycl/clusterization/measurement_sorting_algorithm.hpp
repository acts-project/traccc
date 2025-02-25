/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <functional>

namespace traccc::sycl {

/// Algorithm sorting the reconstructed measurements in their container
///
/// The track finding algorithm expects measurements belonging to a single
/// detector module to be consecutive in memory. But
/// @c traccc::sycl::clusterization_algorithm does not (currently) produce the
/// measurements in such an ordered state. This is where this algorithm comes
/// to the rescue.
///
class measurement_sorting_algorithm
    : public algorithm<measurement_collection_types::view(
          const measurement_collection_types::view&)>,
      public messaging {

    public:
    /// Constructor for the algorithm
    ///
    /// @param copy The copy object to use in the algorithm
    /// @param queue Wrapper for the for the SYCL queue for kernel invocation
    ///
    measurement_sorting_algorithm(
        vecmem::copy& copy, queue_wrapper& queue,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    /// Callable operator performing the sorting on a container
    ///
    /// @param measurements The measurements to sort
    ///
    output_type operator()(const measurement_collection_types::view&
                               measurements_view) const override;

    private:
    /// Copy object to use in the algorithm
    std::reference_wrapper<vecmem::copy> m_copy;
    /// The SYCL queue to use
    std::reference_wrapper<queue_wrapper> m_queue;
};  // class measurement_sorting_algorithm

}  // namespace traccc::sycl
