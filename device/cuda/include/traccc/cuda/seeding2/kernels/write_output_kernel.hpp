/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <traccc/cuda/seeding2/types/internal_sp.hpp>
#include <traccc/edm/alt_seed.hpp>
#include <traccc/edm/internal_spacepoint.hpp>
#include <traccc/edm/seed.hpp>
#include <traccc/edm/spacepoint.hpp>
#include <traccc/utils/memory_resource.hpp>

namespace traccc::cuda {
/**
 * @brief Kernel to write output data back into traccc's EDM.
 *
 * @return A vector buffer containing the output seeds.
 */
alt_seed_collection_types::buffer write_output(const traccc::memory_resource &,
                                               uint32_t, const internal_sp_t,
                                               const alt_seed *const);
}  // namespace traccc::cuda
