/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/alpaka/utils/queue.hpp"

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/vector.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <functional>

namespace traccc::alpaka {

/// Algorithm sorting the reconstructed measurements in their container
///
/// The track finding algorithm expects measurements belonging to a single
/// detector module to be consecutive in memory. But
/// @c traccc::alpaka::clusterization_algorithm does not (currently) produce the
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
    /// @param q The Alpaka queue to schedule the measurement sorting in
    ///
    measurement_sorting_algorithm(
        const traccc::memory_resource& mr, ::vecmem::copy& copy, queue& q,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    /// Callable operator performing the sorting on a container
    ///
    /// @param measurements The measurements to sort
    ///
    output_type operator()(const measurement_collection_types::view&
                               measurements_view) const override;

    private:
    // The memory resource(s) to use
    traccc::memory_resource m_mr;
    /// Copy object to use in the algorithm
    std::reference_wrapper<::vecmem::copy> m_copy;
    /// The Alpaka queue to use
    std::reference_wrapper<queue> m_queue;

};  // class measurement_sorting_algorithm

}  // namespace traccc::alpaka
