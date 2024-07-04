/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/alpaka/clusterization/clusterization_algorithm.hpp"

#include "../utils/barrier.hpp"
#include "../utils/utils.hpp"

// Project include(s)
#include "traccc/clusterization/device/ccl_kernel.hpp"

// System include(s).
#include <algorithm>

namespace traccc::alpaka {

struct CCLKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const cell_collection_types::const_view cells_view,
        const cell_module_collection_types::const_view modules_view,
        const device::details::index_t max_cells_per_partition,
        const device::details::index_t target_cells_per_partition,
        measurement_collection_types::view measurements_view,
        vecmem::data::vector_view<unsigned int> cell_links) const {

        auto const localThreadIdx =
            ::alpaka::getIdx<::alpaka::Block, ::alpaka::Threads>(acc)[0u];
        auto const localBlockIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Blocks>(acc)[0u];
        auto const blockExtent =
            ::alpaka::getWorkDiv<::alpaka::Block, ::alpaka::Threads>(acc)[0u];

        auto& partition_start =
            ::alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
        auto& partition_end =
            ::alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
        auto& outi = ::alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);

        device::details::index_t* const shared_v =
            ::alpaka::getDynSharedMem<device::details::index_t>(acc);
        vecmem::data::vector_view<device::details::index_t> f_view{
            max_cells_per_partition, shared_v};
        vecmem::data::vector_view<device::details::index_t> gf_view{
            max_cells_per_partition, shared_v + max_cells_per_partition};

        alpaka::barrier<TAcc> barry_r(&acc);

        device::ccl_kernel(localThreadIdx, blockExtent, localBlockIdx,
                           cells_view, modules_view, max_cells_per_partition,
                           target_cells_per_partition, partition_start,
                           partition_end, outi, f_view, gf_view, barry_r,
                           measurements_view, cell_links);
    }
};

clusterization_algorithm::clusterization_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy,
    const unsigned short target_cells_per_partition)
    : m_target_cells_per_partition(target_cells_per_partition),
      m_mr(mr),
      m_copy(copy) {}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const cell_collection_types::const_view& cells,
    const cell_module_collection_types::const_view& modules) const {

    // Setup alpaka
    auto devAcc = ::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, 0u);
    auto queue = Queue{devAcc};

    // Number of cells
    const cell_collection_types::view::size_type num_cells =
        m_copy.get().get_size(cells);

    // Create the result object, overestimating the number of measurements.
    measurement_collection_types::buffer measurements{
        num_cells, m_mr.main, vecmem::data::buffer_type::resizable};
    m_copy.get().setup(measurements)->ignore();

    // If there are no cells, return right away.
    if (num_cells == 0) {
        return measurements;
    }

    // Create buffer for linking cells to their measurements.
    //
    // @todo Construct cell clusters on demand in a member function for
    // debugging.
    //
    vecmem::data::vector_buffer<unsigned int> cell_links(num_cells, m_mr.main);
    m_copy.get().setup(cell_links)->ignore();

    // Launch ccl kernel. Each thread will handle a single cell.
    const device::details::ccl_kernel_helper helper{
        m_target_cells_per_partition, num_cells};
    auto workDiv =
        makeWorkDiv<Acc>(helper.num_partitions, helper.threads_per_partition);

    ::alpaka::exec<Acc>(
        queue, workDiv, CCLKernel{}, cells, modules,
        helper.max_cells_per_partition, m_target_cells_per_partition,
        vecmem::get_data(measurements), vecmem::get_data(cell_links));
    ::alpaka::wait(queue);

    return measurements;
}

}  // namespace traccc::alpaka

// Define the required trait needed for Dynamic shared memory allocation.
namespace alpaka::trait {

template <typename TAcc>
struct BlockSharedMemDynSizeBytes<traccc::alpaka::CCLKernel, TAcc> {
    template <typename TVec, typename... TArgs>
    ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
        traccc::alpaka::CCLKernel const& /* kernel */,
        TVec const& /* blockThreadExtent */, TVec const& /* threadElemExtent */,
        const traccc::cell_collection_types::const_view /* cells_view */,
        const traccc::cell_module_collection_types::
            const_view /* modules_view */,
        const unsigned short max_cells_per_partition, TArgs const&... /* args */
        ) -> std::size_t {
        return static_cast<std::size_t>(2 * max_cells_per_partition *
                                        sizeof(unsigned short));
    }
};

}  // namespace alpaka::trait
