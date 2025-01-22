/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

namespace traccc::host {

/// Algorithm sorting the reconstructed measurements in their container
///
/// The track finding algorithm expects measurements belonging to a single
/// detector module to be consecutive in memory. But certain clusterization /
/// measurement creation algorithms may not produce the measurements in such
/// an ordered state. In such cases this algorithm can be used to sort the
/// measurements "correctly" in place.
///
class measurement_sorting_algorithm
    : public algorithm<measurement_collection_types::view(
          const measurement_collection_types::view&)>,
      public messaging {

    public:
    /// Callable operator performing the sorting on a container
    ///
    /// @param measurements The measurements to sort
    ///
    output_type operator()(const measurement_collection_types::view&
                               measurements_view) const override;
};  // class measurement_sorting_algorithm

}  // namespace traccc::host
