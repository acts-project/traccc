/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <traccc/edm/alt_seed.hpp>
#include <traccc/edm/spacepoint.hpp>
#include <traccc/seeding/detail/seeding_config.hpp>
#include <traccc/utils/algorithm.hpp>
#include <traccc/utils/memory_resource.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>

namespace traccc::cuda {
/**
 * @brief Alternative seed finding algorithm, using orthogonal range search
 * implemented through a k-d tree.
 */
class seed_finding2 : public algorithm<alt_seed_collection_types::buffer(
                          const spacepoint_collection_types::const_view&)> {
    public:
    seed_finding2(const traccc::memory_resource& mr);

    output_type operator()(
        const spacepoint_collection_types::const_view& sps) const override;

    private:
    traccc::memory_resource m_output_mr;
    seedfinder_config m_finder_conf;
    seedfilter_config m_filter_conf;
};
}  // namespace traccc::cuda
