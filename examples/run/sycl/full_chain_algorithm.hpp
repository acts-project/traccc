/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/device/container_h2d_copy_alg.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/sycl/clusterization/clusterization_algorithm.hpp"
#include "traccc/sycl/seeding/seeding_algorithm.hpp"
#include "traccc/sycl/seeding/track_params_estimation.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/memory/sycl/device_memory_resource.hpp>
#include <vecmem/utils/sycl/copy.hpp>

namespace traccc::sycl {
namespace details {
/// Internal data type used by @c traccc::sycl::full_chain_algorithm
struct full_chain_algorithm_data;
}  // namespace details

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
    /// Private data object
    details::full_chain_algorithm_data* m_data;
    /// Device memory resource
    vecmem::sycl::device_memory_resource m_device_mr;
    /// Memory copy object
    mutable vecmem::sycl::copy m_copy;

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

}  // namespace traccc::sycl
