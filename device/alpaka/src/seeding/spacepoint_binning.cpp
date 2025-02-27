/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/alpaka/seeding/details/spacepoint_binning.hpp"

#include "../utils/utils.hpp"

// Project include(s).
#include "traccc/seeding/device/count_grid_capacities.hpp"
#include "traccc/seeding/device/populate_grid.hpp"

namespace traccc::alpaka {
namespace kernels {

// Grid Capacity Kernel
struct CountGridCapacity {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const seedfinder_config& config,
        const traccc::details::spacepoint_grid_types::host::axis_p0_type
            phi_axis,
        const traccc::details::spacepoint_grid_types::host::axis_p1_type z_axis,
        const edm::spacepoint_collection::const_view spacepoints_view,
        vecmem::data::vector_view<unsigned int> grid_capacities_view) const {
        auto const globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u];

        device::count_grid_capacities(globalThreadIdx, config, phi_axis, z_axis,
                                      spacepoints_view, grid_capacities_view);
    }
};

// Populate Grid Kernel
struct PopulateGrid {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, seedfinder_config config,
        edm::spacepoint_collection::const_view spacepoints_view,
        traccc::details::spacepoint_grid_types::view grid_view) const {
        auto const globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u];

        device::populate_grid(globalThreadIdx, config, spacepoints_view,
                              grid_view);
    }
};

}  // namespace kernels

namespace details {

spacepoint_binning::spacepoint_binning(
    const seedfinder_config& config, const spacepoint_grid_config& grid_config,
    const traccc::memory_resource& mr, vecmem::copy& copy,
    std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)),
      m_config(config),
      m_axes(get_axes(grid_config, (mr.host ? *(mr.host) : mr.main))),
      m_mr(mr),
      m_copy(copy) {}

traccc::details::spacepoint_grid_types::buffer spacepoint_binning::operator()(
    const edm::spacepoint_collection::const_view& spacepoints_view) const {

    // Setup alpaka
    auto const platformAcc = ::alpaka::Platform<Acc>{};
    auto devAcc = ::alpaka::getDevByIdx(platformAcc, 0u);
    auto queue = Queue{devAcc};

    // Get the spacepoint sizes from the view
    auto sp_size = m_copy.get_size(spacepoints_view);

    if (sp_size == 0) {
        return {m_axes.first, m_axes.second, {}, m_mr.main, m_mr.host};
    }

    // Set up the container that will be filled with the required capacities for
    // the spacepoint grid.
    const unsigned int grid_bins = m_axes.first.n_bins * m_axes.second.n_bins;
    vecmem::data::vector_buffer<unsigned int> grid_capacities_buff(grid_bins,
                                                                   m_mr.main);
    m_copy.setup(grid_capacities_buff)->ignore();
    m_copy.memset(grid_capacities_buff, 0)->ignore();
    vecmem::data::vector_view<unsigned int> grid_capacities_view =
        grid_capacities_buff;

    // Now define the Alpaka Work division
    auto const deviceProperties = ::alpaka::getAccDevProps<Acc>(devAcc);
    auto const maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];
    auto const threadsPerBlock = maxThreadsPerBlock;
    auto const blocksPerGrid =
        (sp_size + threadsPerBlock - 1) / threadsPerBlock;
    auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

    ::alpaka::exec<Acc>(queue, workDiv, kernels::CountGridCapacity{}, m_config,
                        m_axes.first, m_axes.second, spacepoints_view,
                        grid_capacities_view);
    ::alpaka::wait(queue);

    // Copy grid capacities back to the host
    vecmem::vector<unsigned int> grid_capacities_host(m_mr.host ? m_mr.host
                                                                : &(m_mr.main));
    m_copy(grid_capacities_buff, grid_capacities_host)->wait();

    // Create the grid buffer.
    traccc::details::spacepoint_grid_types::buffer grid_buffer(
        m_axes.first, m_axes.second,
        std::vector<std::size_t>(grid_capacities_host.begin(),
                                 grid_capacities_host.end()),
        m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable);
    m_copy.setup(grid_buffer._buffer)->wait();
    traccc::details::spacepoint_grid_types::view grid_view = grid_buffer;

    ::alpaka::exec<Acc>(queue, workDiv, kernels::PopulateGrid{}, m_config,
                        spacepoints_view, grid_view);
    ::alpaka::wait(queue);

    // Return the freshly filled buffer.
    return grid_buffer;
}

}  // namespace details
}  // namespace traccc::alpaka
