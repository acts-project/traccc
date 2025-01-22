/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/alpaka/clusterization/clusterization_algorithm.hpp"
#include "traccc/alpaka/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/alpaka/clusterization/spacepoint_formation_algorithm.hpp"
#include "traccc/alpaka/seeding/seeding_algorithm.hpp"
#include "traccc/alpaka/seeding/track_params_estimation.hpp"
#include "traccc/device/container_h2d_copy_alg.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
#include <vecmem/memory/hip/device_memory_resource.hpp>
#include <vecmem/utils/hip/copy.hpp>
#endif

#include <vecmem/memory/binary_page_memory_resource.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <memory>

namespace traccc::alpaka {

/// Algorithm performing the full chain of track reconstruction
///
/// At least as much as is implemented in the project at any given moment.
///
class full_chain_algorithm
    : public algorithm<bound_track_parameters_collection_types::host(
          const cell_collection_types::host&,
          const cell_module_collection_types::host&)>,
      public messaging {

    public:
    /// Algorithm constructor
    ///
    /// @param mr The memory resource to use for the intermediate and result
    ///           objects
    /// @param target_cells_per_partition The average number of cells in each
    /// partition.
    ///
    full_chain_algorithm(vecmem::memory_resource& host_mr,
                         const unsigned short target_cells_per_partiton,
                         const seedfinder_config& finder_config,
                         const spacepoint_grid_config& grid_config,
                         const seedfilter_config& filter_config);

    /// Copy constructor
    ///
    /// An explicit copy constructor is necessary because in the MT tests
    /// we do want to copy such objects, but a default copy-constructor can
    /// not be generated for them.
    ///
    /// @param parent The parent algorithm chain to copy
    ///
    full_chain_algorithm(const full_chain_algorithm& parent);

    /// Algorithm destructor
    ~full_chain_algorithm();

    /// Reconstruct track parameters in the entire detector
    ///
    /// @param cells The cells for every detector module in the event
    /// @return The track parameters reconstructed
    ///
    output_type operator()(
        const cell_collection_types::host& cells,
        const cell_module_collection_types::host& modules) const override;

    private:
    /// Host memory resource
    vecmem::memory_resource& m_host_mr;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    /// Device memory resource
    vecmem::cuda::device_memory_resource m_device_mr;
    /// Memory copy object
    vecmem::cuda::copy m_copy;
#elif ALPAKA_ACC_GPU_HIP_ENABLED
    /// Device memory resource
    vecmem::hip::device_memory_resource m_device_mr;
    /// Memory copy object
    vecmem::hip::copy m_copy;
#else
    /// Device memory resource
    vecmem::memory_resource& m_device_mr;
    /// Memory copy object
    vecmem::copy m_copy;
#endif
    /// Device caching memory resource
    std::unique_ptr<vecmem::binary_page_memory_resource> m_cached_device_mr;

    /// @name Sub-algorithms used by this full-chain algorithm
    /// @{

    /// The average number of cells in each partition.
    /// Adapt to different GPUs' capabilities.
    unsigned short m_target_cells_per_partition;
    /// Clusterization algorithm
    clusterization_algorithm m_clusterization;
    /// Measurement sorting algorithm
    measurement_sorting_algorithm m_measurement_sorting;
    /// Spacepoint formation algorithm
    spacepoint_formation_algorithm m_spacepoint_formation;
    /// Seeding algorithm
    seeding_algorithm m_seeding;
    /// Track parameter estimation algorithm
    track_params_estimation m_track_parameter_estimation;

    /// Configs
    seedfinder_config m_finder_config;
    spacepoint_grid_config m_grid_config;
    seedfilter_config m_filter_config;

    /// @}

};  // class full_chain_algorithm

}  // namespace traccc::alpaka
