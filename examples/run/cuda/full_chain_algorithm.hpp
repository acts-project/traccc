/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/device/container_h2d_copy_alg.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/binary_page_memory_resource.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// System include(s).
#include <memory>

namespace traccc::cuda {

/// Algorithm performing the full chain of track reconstruction
///
/// At least as much as is implemented in the project at any given moment.
///
class full_chain_algorithm
    : public algorithm<bound_track_parameters_collection_types::host(
          const cell_container_types::host&)> {

    public:
    /// Algorithm constructor
    ///
    /// @param mr The memory resource to use for the intermediate and result
    ///           objects
    ///
    full_chain_algorithm(vecmem::memory_resource& host_mr);

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
        const cell_container_types::host& cells) const override;

    private:
    /// Host memory resource
    vecmem::memory_resource& m_host_mr;
    /// Device memory resource
    vecmem::cuda::device_memory_resource m_device_mr;
    /// Device caching memory resource
    std::unique_ptr<vecmem::binary_page_memory_resource> m_cached_device_mr;
    /// Memory copy object
    mutable vecmem::cuda::copy m_copy;

    /// @name Sub-algorithms used by this full-chain algorithm
    /// @{

    /// Host->Device cell copy algorithm
    device::container_h2d_copy_alg<cell_container_types> m_host2device;
    /// Clusterization algorithm
    clusterization_algorithm m_clusterization;
    /// Seeding algorithm
    seeding_algorithm m_seeding;
    /// Track parameter estimation algorithm
    track_params_estimation m_track_parameter_estimation;

    /// @}

};  // class full_chain_algorithm

}  // namespace traccc::cuda
