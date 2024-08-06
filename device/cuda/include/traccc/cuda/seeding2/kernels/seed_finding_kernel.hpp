/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <traccc/cuda/seeding2/types/kd_tree.hpp>
#include <traccc/edm/internal_spacepoint.hpp>
#include <traccc/edm/seed.hpp>
#include <traccc/edm/spacepoint.hpp>
#include <traccc/seeding/detail/seeding_config.hpp>
#include <vecmem/utils/copy.hpp>

namespace traccc::cuda {
/**
 * @brief Execute the seed finding kernel itself.
 *
 * @return A pair containing the list of internal seeds as well as the number
 * of seeds.
 */
seed_collection_types::buffer run_seeding(
    seedfinder_config, seedfilter_config, vecmem::memory_resource &,
    vecmem::copy &, const spacepoint_collection_types::const_view &, kd_tree_t);
}  // namespace traccc::cuda
