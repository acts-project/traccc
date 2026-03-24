/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/cuda/utils/algorithm_base.hpp"

// Project include(s).
#include "traccc/device/algorithm_base.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/messaging.hpp"

namespace traccc::cuda {

/// Algorithm sorting the detector cell information
///
/// Clusterization algorithms may (and do) rely on cells being "strictly
/// sorted". This algorithm can be used to make sure that this would be the
/// case.
///
class silicon_cell_sorting_algorithm
    : public algorithm<edm::silicon_cell_collection::buffer(
          const edm::silicon_cell_collection::const_view&)>,
      device::algorithm_base,
      cuda::algorithm_base,
      public messaging {

    public:
    /// Constructor for the algorithm
    ///
    /// @param mr The memory resource(s) to use in the algorithm
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param str The CUDA stream to perform the operations in
    /// @param config The clustering configuration
    ///
    silicon_cell_sorting_algorithm(
        const traccc::memory_resource& mr, ::vecmem::copy& copy,
        cuda::stream& str,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    /// Callable operator performing the sorting on a container
    ///
    /// @param cells The cells to sort
    /// @return The sorted cells
    ///
    output_type operator()(
        const edm::silicon_cell_collection::const_view& cells) const override;

};  // class silicon_cell_sorting_algorithm

}  // namespace traccc::cuda
