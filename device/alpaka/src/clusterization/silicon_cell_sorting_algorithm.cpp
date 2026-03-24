/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/alpaka/clusterization/silicon_cell_sorting_algorithm.hpp"

#include "../utils/get_queue.hpp"
#include "../utils/parallel_algorithms.hpp"
#include "../utils/thread_id.hpp"

// Project include(s).
#include "traccc/clusterization/device/silicon_cell_sorter.hpp"
#include "traccc/clusterization/device/sorting_index_filler.hpp"

namespace traccc::alpaka {
namespace kernels {

/// Kernel filling the output buffer with sorted cells.
struct fill_sorted_silicon_cells {
    /// @param[in] acc Alpaka accelerator object
    /// @param[in] input_view View of the input cells
    /// @param[out] output_view View of the output cells
    /// @param[in] sorted_indices_view View of the sorted cell indices
    ///
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        const edm::silicon_cell_collection::const_view input_view,
        edm::silicon_cell_collection::view output_view,
        const vecmem::data::vector_view<const unsigned int> sorted_indices_view)
        const {

        // Create the device objects.
        const edm::silicon_cell_collection::const_device input{input_view};
        edm::silicon_cell_collection::device output{output_view};
        const vecmem::device_vector<const unsigned int> sorted_indices{
            sorted_indices_view};

        // Stop early if we can.
        const unsigned int index = details::thread_id1{acc}.getGlobalThreadId();
        if (index >= input.size()) {
            return;
        }

        // Copy one measurement into the correct position.
        output.at(index) = input.at(sorted_indices.at(index));
    }
};  // struct fill_sorted_silicon_cells

}  // namespace kernels

silicon_cell_sorting_algorithm::silicon_cell_sorting_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy, alpaka::queue& q,
    std::unique_ptr<const Logger> logger)
    : device::algorithm_base(mr, copy),
      alpaka::algorithm_base(q),
      messaging(std::move(logger)) {}

auto silicon_cell_sorting_algorithm::operator()(
    const edm::silicon_cell_collection::const_view& cells_view) const
    -> output_type {

    // Exit early if there are no cells.
    if (cells_view.capacity() == 0) {
        return {};
    }

    // Get a convenience variable for the queue that we'll be using.
    auto& aqueue = details::get_queue(queue());

    // Create a vector of cell indices, which would be sorted.
    vecmem::data::vector_buffer<unsigned int> indices(cells_view.capacity(),
                                                      mr().main);
    copy().setup(indices)->wait();
    details::for_each(aqueue, mr(), indices.ptr(),
                      indices.ptr() + indices.capacity(),
                      device::sorting_index_filler{indices});

    // Sort the indices according to the (correct) order of the cells.
    details::sort(aqueue, mr(), indices.ptr(),
                  indices.ptr() + indices.capacity(),
                  device::silicon_cell_sorter{cells_view});

    // Create the output buffer.
    output_type result{cells_view.capacity(), mr().main,
                       cells_view.size().capacity()
                           ? vecmem::data::buffer_type::resizable
                           : vecmem::data::buffer_type::fixed_size};
    copy().setup(result)->ignore();
    copy()(cells_view.size(), result.size())->ignore();

    // Fill it with the sorted cells.
    const unsigned int BLOCK_SIZE = warp_size() * 8;
    const unsigned int n_blocks =
        (cells_view.capacity() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    auto workDiv = makeWorkDiv<Acc>(BLOCK_SIZE, n_blocks);
    ::alpaka::exec<Acc>(aqueue, workDiv, kernels::fill_sorted_silicon_cells{},
                        cells_view, vecmem::get_data(result),
                        vecmem::get_data(indices));

    // Return the sorted buffer.
    return result;
}

}  // namespace traccc::alpaka
