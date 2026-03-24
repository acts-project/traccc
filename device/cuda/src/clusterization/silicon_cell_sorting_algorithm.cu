/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/thread_id.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/clusterization/silicon_cell_sorting_algorithm.hpp"

// Project include(s).
#include "traccc/clusterization/device/silicon_cell_sorter.hpp"
#include "traccc/clusterization/device/sorting_index_filler.hpp"

// Thrust include(s).
#include <thrust/for_each.h>
#include <thrust/sort.h>

namespace traccc::cuda {
namespace kernels {

/// Kernel filling the output buffer with sorted cells.
///
/// @param[in] input_view View of the input cells
/// @param[out] output_view View of the output cells
/// @param[in] sorted_indices_view View of the sorted cell indices
///
__global__ void fill_sorted_silicon_cells(
    const edm::silicon_cell_collection::const_view input_view,
    edm::silicon_cell_collection::view output_view,
    const vecmem::data::vector_view<const unsigned int> sorted_indices_view) {

    // Create the device objects.
    const edm::silicon_cell_collection::const_device input{input_view};
    edm::silicon_cell_collection::device output{output_view};
    const vecmem::device_vector<const unsigned int> sorted_indices{
        sorted_indices_view};

    // Stop early if we can.
    const unsigned int index = details::thread_id1{}.getGlobalThreadId();
    if (index >= input.size()) {
        return;
    }

    // Copy one measurement into the correct position.
    output.at(index) = input.at(sorted_indices.at(index));
}

}  // namespace kernels

silicon_cell_sorting_algorithm::silicon_cell_sorting_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy, cuda::stream& str,
    std::unique_ptr<const Logger> logger)
    : device::algorithm_base(mr, copy),
      cuda::algorithm_base(str),
      messaging(std::move(logger)) {}

auto silicon_cell_sorting_algorithm::operator()(
    const edm::silicon_cell_collection::const_view& cells_view) const
    -> output_type {

    // Exit early if there are no cells.
    if (cells_view.capacity() == 0) {
        return {};
    }

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t cstream = details::get_stream(stream());
    // Set up the Thrust execution policy.
    auto policy =
        thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(mr().main)))
            .on(cstream);

    // Create a vector of cell indices, which would be sorted.
    vecmem::data::vector_buffer<unsigned int> indices(cells_view.capacity(),
                                                      mr().main);
    copy().setup(indices)->wait();
    thrust::for_each(policy, indices.ptr(), indices.ptr() + indices.capacity(),
                     device::sorting_index_filler{indices});

    // Sort the indices according to the (correct) order of the cells.
    thrust::sort(policy, indices.ptr(), indices.ptr() + indices.capacity(),
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
    kernels::fill_sorted_silicon_cells<<<n_blocks, BLOCK_SIZE, 0, cstream>>>(
        cells_view, result, indices);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    // Return the sorted buffer.
    return result;
}

}  // namespace traccc::cuda
