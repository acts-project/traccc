/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <traccc/edm/seed.hpp>
#include <traccc/edm/spacepoint.hpp>
#include <traccc/seeding/detail/seeding_config.hpp>
#include <traccc/utils/algorithm.hpp>
#include <traccc/utils/memory_resource.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

#include "traccc/cuda/utils/stream.hpp"

namespace traccc::cuda {
/**
 * @brief Alternative seed finding algorithm, using orthogonal range search
 * implemented through a k-d tree.
 */
class seed_finding2 : public algorithm<seed_collection_types::buffer(
                          const spacepoint_collection_types::const_view&)> {
    public:
    seed_finding2(const seedfinder_config& config,
                  const seedfilter_config& filter_config,
                  const traccc::memory_resource& mr, vecmem::copy& copy,
                  stream& str);

    output_type operator()(
        const spacepoint_collection_types::const_view& sps) const override;

    private:
    traccc::memory_resource m_mr;
    vecmem::copy& m_copy;
    stream& m_stream;

    seedfinder_config m_seedfinder_config;
    seedfilter_config m_seedfilter_config;

    int m_warp_size;
};
}  // namespace traccc::cuda
