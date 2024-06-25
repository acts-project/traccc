/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <traccc/cuda/seeding2/types/kd_tree.hpp>
#include <traccc/edm/internal_spacepoint.hpp>
#include <traccc/edm/spacepoint.hpp>
#include <vecmem/utils/copy.hpp>
#include <vector>

namespace traccc::cuda {
/**
 * @brief Creates a k-d tree from a given set of spacepoints.
 *
 * @return A pair containing the k-d tree nodes as well as the number of nodes.
 */
std::tuple<kd_tree_owning_t, uint32_t, vecmem::data::vector_buffer<std::size_t>>
create_kd_tree(vecmem::memory_resource&, vecmem::copy& copy,
               const spacepoint_collection_types::const_view&);
}  // namespace traccc::cuda
