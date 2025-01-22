/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"

namespace traccc::host {

silicon_pixel_spacepoint_formation_algorithm::
    silicon_pixel_spacepoint_formation_algorithm(
        vecmem::memory_resource& mr, std::unique_ptr<const Logger> logger)
    : m_mr(mr), m_logger(std::move(logger)) {}

}  // namespace traccc::host
