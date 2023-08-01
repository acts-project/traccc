/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/alpaka/utils/definitions.hpp"
#include "traccc/alpaka/utils/barrier.hpp"
#include "traccc/alpaka/clusterization/clusterization_algorithm.hpp"

// Project include(s)
#include "traccc/clusterization/device/aggregate_cluster.hpp"
#include "traccc/clusterization/device/ccl_kernel.hpp"
#include "traccc/clusterization/device/form_spacepoints.hpp"
#include "traccc/clusterization/device/reduce_problem_cell.hpp"

// System include(s).
#include <algorithm>

namespace traccc::alpaka {

namespace {
/// These indices in clusterization will only range from 0 to
/// max_cells_per_partition, so we only need a short.
using index_t = unsigned short;

static constexpr int TARGET_CELLS_PER_THREAD = 8;
static constexpr int MAX_CELLS_PER_THREAD = 12;
}  // namespace

struct CCLKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        const cell_collection_types::const_view cells_view,
        const cell_module_collection_types::const_view modules_view,
        const index_t max_cells_per_partition,
        const index_t target_cells_per_partition,
        alt_measurement_collection_types::view measurements_view,
        unsigned int* measurement_count,
        vecmem::data::vector_view<unsigned int> cell_links
    ) const {

        index_t const localThreadIdx =
            ::alpaka::getIdx<::alpaka::Block, ::alpaka::Threads>(acc)[0u];
        unsigned int const localBlockIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Blocks>(acc)[0u];
        index_t const blockExtent =
            ::alpaka::getWorkDiv<::alpaka::Grid, ::alpaka::Blocks>(acc)[0u];

        unsigned int partition_start = ::alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
        unsigned int partition_end = ::alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
        unsigned int outi = ::alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);

        index_t* const shared_v = ::alpaka::getDynSharedMem<index_t>(acc);
        index_t* f = &shared_v[0];
        index_t* f_next = &shared_v[max_cells_per_partition];

        alpaka::barrier<TAcc> barry_r(acc);

        device::ccl_kernel(localThreadIdx, blockExtent, localBlockIdx, cells_view,
                           modules_view, max_cells_per_partition,
                           target_cells_per_partition, partition_start,
                           partition_end, outi, f, f_next, barry_r,
                           measurements_view, *measurement_count, cell_links);
    }
};

struct FormSpacepointsKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alt_measurement_collection_types::const_view measurements_view,
        cell_module_collection_types::const_view modules_view,
        const unsigned int* measurement_count,
        spacepoint_collection_types::view spacepoints_view
    ) const {

        auto const globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u];

        device::form_spacepoints(globalThreadIdx, measurements_view,
                                 modules_view, *measurement_count,
                                 spacepoints_view);
    }
};

clusterization_algorithm::clusterization_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy,
    const unsigned short target_cells_per_partition)
    : m_mr(mr),
      m_copy(copy),
      m_target_cells_per_partition(target_cells_per_partition) {}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const cell_collection_types::const_view& cells,
    const cell_module_collection_types::const_view& modules) const {

    // Setup alpaka
    auto devHost = ::alpaka::getDevByIdx<Host>(0u);
    auto devAcc = ::alpaka::getDevByIdx<Acc>(0u);
    auto queue = Queue{devAcc};

    // Number of cells
    const cell_collection_types::view::size_type num_cells =
        m_copy.get_size(cells);

    if (num_cells == 0) {
        return {output_type::first_type{0, m_mr.main},
                output_type::second_type{0, m_mr.main}};
    }

    // Create result object for the CCL kernel with size overestimation
    alt_measurement_collection_types::buffer measurements_buffer(num_cells,
                                                                 m_mr.main);
    m_copy.setup(measurements_buffer);

    // Counter for number of measurements
    auto bufHost_num_measurements =
        ::alpaka::allocBuf<unsigned int, Idx>(devHost, 1u);
    unsigned int* num_measurements_host(::alpaka::getPtrNative(bufHost_num_measurements));
    num_measurements_host = 0;
    auto bufAcc_num_measurements =
        ::alpaka::allocBuf<unsigned int, Idx>(devAcc, 1u);
    ::alpaka::memcpy(queue, bufAcc_num_measurements, bufHost_num_measurements);

    const unsigned short max_cells_per_partition =
        (m_target_cells_per_partition * MAX_CELLS_PER_THREAD +
         TARGET_CELLS_PER_THREAD - 1) /
        TARGET_CELLS_PER_THREAD;
    const unsigned int threads_per_partition =
        (m_target_cells_per_partition + TARGET_CELLS_PER_THREAD - 1) /
        TARGET_CELLS_PER_THREAD;
    const unsigned int num_partitions =
        (num_cells + m_target_cells_per_partition - 1) /
        m_target_cells_per_partition;
    auto workDiv = makeWorkDiv<Acc>(num_partitions, threads_per_partition);

    // Create buffer for linking cells to their spacepoints.
    vecmem::data::vector_buffer<unsigned int> cell_links(num_cells, m_mr.main);
    m_copy.setup(cell_links);

    // Launch ccl kernel. Each thread will handle a single cell.
    ::alpaka::exec<Acc>(queue, workDiv, CCLKernel{},
            cells,
            modules,
            max_cells_per_partition,
            m_target_cells_per_partition,
            vecmem::get_data(measurements_buffer),
            ::alpaka::getPtrNative(bufAcc_num_measurements),
            vecmem::get_data(cell_links)
    );
    ::alpaka::wait(queue);

    // Copy number of measurements to host
    ::alpaka::memcpy(queue, bufHost_num_measurements, bufAcc_num_measurements);

    spacepoint_collection_types::buffer spacepoints_buffer(
        *num_measurements_host, m_mr.main);
    m_copy.setup(spacepoints_buffer);

    // For the following kernel, we can now use whatever the desired number of
    // threads per block.
    auto spacepointsLocalSize = 1024;
    const unsigned int num_blocks =
        (*num_measurements_host + spacepointsLocalSize - 1) /
        spacepointsLocalSize;
    workDiv = makeWorkDiv<Acc>(num_blocks, spacepointsLocalSize);

    // Turn 2D measurements into 3D spacepoints
    ::alpaka::exec<Acc>(queue, workDiv, FormSpacepointsKernel{},
        vecmem::get_data(measurements_buffer),
        modules,
        ::alpaka::getPtrNative(bufAcc_num_measurements),
        vecmem::get_data(spacepoints_buffer));
    ::alpaka::wait(queue);

    return {std::move(spacepoints_buffer), std::move(cell_links)};
}

}  // namespace traccc::alpaka

// Define the required trait needed for Dynamic shared memory allocation.
namespace alpaka::trait {

template <typename TAcc>
struct BlockSharedMemDynSizeBytes<traccc::alpaka::CCLKernel, TAcc>
{
    template <typename TVec>
    ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
        traccc::alpaka::CCLKernel const& /* kernel */,
        TVec const& /* blockThreadExtent */,
        TVec const& /* threadElemExtent */,
        const traccc::cell_collection_types::const_view /* cells_view */,
        const traccc::cell_module_collection_types::const_view /* modules_view */,
        const unsigned short max_cells_per_partition,
        const unsigned short /* target_cells_per_partition */,
        traccc::alt_measurement_collection_types::view /* measurements_view */,
        unsigned int* /* measurement_count */,
        vecmem::data::vector_view<unsigned int> /* cell_link */
    ) -> std::size_t {
        return static_cast<std::size_t>(2 * max_cells_per_partition * sizeof(unsigned short));
    }
};

}  // namespace alpaka::traits
