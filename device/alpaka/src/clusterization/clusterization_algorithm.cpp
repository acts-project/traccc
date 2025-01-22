/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/alpaka/clusterization/clusterization_algorithm.hpp"

#include "../utils/barrier.hpp"
#include "../utils/thread_id.hpp"
#include "../utils/utils.hpp"

// Project include(s)
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/clusterization/device/ccl_kernel.hpp"

// System include(s).
#include <algorithm>
#include <mutex>

namespace traccc::alpaka {

struct CCLKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const clustering_config cfg,
        const edm::silicon_cell_collection::const_view cells_view,
        const silicon_detector_description::const_view det_descr_view,
        vecmem::data::vector_view<device::details::index_t> f_backup_view,
        vecmem::data::vector_view<device::details::index_t> gf_backup_view,
        vecmem::data::vector_view<unsigned char> adjc_backup_view,
        vecmem::data::vector_view<device::details::index_t> adjv_backup_view,
        uint32_t* backup_mutex_ptr,
        measurement_collection_types::view measurements_view,
        vecmem::data::vector_view<unsigned int> cell_links) const {

        details::thread_id1 thread_id(acc);

        auto& partition_start =
            ::alpaka::declareSharedVar<std::size_t, __COUNTER__>(acc);
        auto& partition_end =
            ::alpaka::declareSharedVar<std::size_t, __COUNTER__>(acc);
        auto& outi = ::alpaka::declareSharedVar<std::size_t, __COUNTER__>(acc);

        device::details::index_t* const shared_v =
            ::alpaka::getDynSharedMem<device::details::index_t>(acc);
        vecmem::data::vector_view<device::details::index_t> f_view{
            cfg.max_partition_size(), shared_v};
        vecmem::data::vector_view<device::details::index_t> gf_view{
            cfg.max_partition_size(), shared_v + cfg.max_partition_size()};

        vecmem::device_atomic_ref<uint32_t> backup_mutex(*backup_mutex_ptr);

        alpaka::barrier<TAcc> barry_r(&acc);

        device::ccl_kernel(cfg, thread_id, cells_view, det_descr_view,
                           partition_start, partition_end, outi, f_view,
                           gf_view, f_backup_view, gf_backup_view,
                           adjc_backup_view, adjv_backup_view, backup_mutex,
                           barry_r, measurements_view, cell_links);
    }
};

struct ZeroMutexKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const&, uint32_t* ptr) const {
        *ptr = 0;
    }
};

clusterization_algorithm::clusterization_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy,
    const config_type& config, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)),
      m_config(config),
      m_mr(mr),
      m_copy(copy),
      m_f_backup(m_config.backup_size(), m_mr.main),
      m_gf_backup(m_config.backup_size(), m_mr.main),
      m_adjc_backup(m_config.backup_size(), m_mr.main),
      m_adjv_backup(m_config.backup_size() * 8, m_mr.main),
      m_backup_mutex(vecmem::make_unique_alloc<unsigned int>(m_mr.main)) {}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const edm::silicon_cell_collection::const_view& cells,
    const silicon_detector_description::const_view& det_descr) const {

    // Setup alpaka
    auto devAcc = ::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, 0u);
    auto queue = Queue{devAcc};

    // Setup the mutex, if it is not already setup.
    std::call_once(m_setup_once, [&queue, mutex_ptr = m_backup_mutex.get()]() {
        auto workDiv = makeWorkDiv<Acc>(1, 1);
        ::alpaka::exec<Acc>(queue, workDiv, ZeroMutexKernel{}, mutex_ptr);
        ::alpaka::wait(queue);
    });

    // Number of cells
    const edm::silicon_cell_collection::view::size_type num_cells =
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
    Idx num_blocks = (num_cells + (m_config.target_partition_size()) - 1) /
                     m_config.target_partition_size();
    static_assert(::alpaka::isMultiThreadAcc<Acc>,
                  "Clustering algorithm must be compiled for an accelerator "
                  "with support for multi-thread blocks.");
    auto workDiv = makeWorkDiv<Acc>(num_blocks, m_config.threads_per_partition);

    ::alpaka::exec<Acc>(
        queue, workDiv, CCLKernel{}, m_config, cells, det_descr,
        vecmem::get_data(m_f_backup), vecmem::get_data(m_gf_backup),
        vecmem::get_data(m_adjc_backup), vecmem::get_data(m_adjv_backup),
        m_backup_mutex.get(), vecmem::get_data(measurements),
        vecmem::get_data(cell_links));
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
        const traccc::clustering_config config, TArgs const&... /* args */
        ) -> std::size_t {
        return static_cast<std::size_t>(2 * config.max_partition_size() *
                                        sizeof(unsigned short));
    }
};

}  // namespace alpaka::trait
