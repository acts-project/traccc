/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <functional>
#include <memory>

namespace traccc::host {

/// Algorithm sorting the detector cell information
///
/// Clusterization algorithms may (and do) rely on cells being "strictly
/// sorted". This algorithm can be used to make sure that this would be the
/// case.
///
class silicon_cell_sorting_algorithm
    : public algorithm<edm::silicon_cell_collection::host(
          const edm::silicon_cell_collection::const_view&)>,
      public messaging {

    public:
    /// Constructor
    ///
    /// @param mr The memory resource to use for the algorithm
    /// @param logger The logger to use for the algorithm
    ///
    silicon_cell_sorting_algorithm(
        vecmem::memory_resource& mr,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    /// Callable operator performing the sorting on a container
    ///
    /// @param cells The cells to sort
    /// @return The sorted cells
    ///
    output_type operator()(
        const edm::silicon_cell_collection::const_view& cells) const override;

    private:
    /// The memory resource to use
    std::reference_wrapper<vecmem::memory_resource> m_mr;

};  // class silicon_cell_sorting_algorithm

}  // namespace traccc::host
