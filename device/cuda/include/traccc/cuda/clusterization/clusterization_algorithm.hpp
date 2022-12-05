/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/cuda/utils/stream.hpp"

// Project include(s).
#include "traccc/edm/cell.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

namespace traccc::cuda {

/// Algorithm performing hit clusterization in a naive way
///
/// This algorithm implements a very trivial parallelization for the hit
/// clusterization. Simply handling every detector module in its own thread.
/// Which is a fairly simple way of translating the single-threaded CPU
/// algorithm, but also a pretty bad algorithm for a GPU.
///
class clusterization_algorithm
    : public algorithm<spacepoint_container_types::buffer(
          const cell_container_types::const_view&)> {

    public:
    /// Constructor for clusterization algorithm
    ///
    /// @param mr The memory resource(s) to use in the algorithm
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param str The CUDA stream to perform the operations in
    ///
    clusterization_algorithm(const traccc::memory_resource& mr,
                             vecmem::copy& copy, stream& str);

    /// Callable operator for clusterization algorithm
    ///
    /// @param cells_per_event is a container with cell modules as headers
    /// and cells as the items
    /// @return a spacepoint container (buffer) - jagged vector of spacepoints
    /// per module.
    output_type operator()(
        const cell_container_types::const_view& cells_view) const override;

    private:
    /// The memory resource(s) to use
    traccc::memory_resource m_mr;
    /// The copy object to use
    vecmem::copy& m_copy;
    /// The CUDA stream to use
    stream& m_stream;
};

}  // namespace traccc::cuda