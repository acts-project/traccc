/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

namespace traccc {
namespace cuda {

/// track parameter estimation for cuda
struct track_params_estimation
    : public algorithm<bound_track_parameters_collection_types::host(
          const spacepoint_container_types::const_view&,
          const seed_collection_types::const_view&)> {

    public:
    /// Constructor for track_params_estimation
    ///
    /// @param mr is the memory resource
    track_params_estimation(const traccc::memory_resource& mr);

    /// Callable operator for track_params_esitmation
    ///
    /// @param spaepoints_view   is the view of the spacepoint container
    /// @param seeds_view        is the view of the seed container
    /// @return                  vector of bound track parameters
    ///
    bound_track_parameters_collection_types::host operator()(
        const spacepoint_container_types::const_view& spacepoints_view,
        const seed_collection_types::const_view& seeds_view) const override;

    private:
    /// Memory resource used by the algorithm
    traccc::memory_resource m_mr;
    /// Copy object used by the algorithm
    std::unique_ptr<vecmem::copy> m_copy;
};

}  // namespace cuda
}  // namespace traccc
