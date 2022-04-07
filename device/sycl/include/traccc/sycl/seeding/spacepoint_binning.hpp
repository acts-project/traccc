/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL library include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/common/seeding_config.hpp"
#include "traccc/seeding/common/spacepoint_grid.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <functional>
#include <utility>

namespace traccc::sycl {

/// Spacepoing binning for sycl
struct spacepoint_binning
    : public algorithm<sp_grid(const host_spacepoint_container&)> {

    spacepoint_binning(const seedfinder_config& config,
                       const spacepoint_grid_config& grid_config,
                       vecmem::memory_resource& mr, const queue_wrapper& queue);

    unsigned int nbins() const;

    output_type operator()(
        const host_spacepoint_container& spacepoints) const override;

    seedfinder_config m_config;
    spacepoint_grid_config m_grid_config;
    std::pair<output_type::axis_p0_type, output_type::axis_p1_type> m_axes;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
    mutable queue_wrapper m_queue;
};

}  // namespace traccc::sycl
