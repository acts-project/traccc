/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"

// Thrust include(s).
#include <thrust/pair.h>

namespace traccc::device {

/// Function filling the module map in an unsorted manner
///
/// @param[in] globalIndex         The index of the current thread
/// @param[in] measurements_view   Measurement container view object
/// @param[out] module_map_view    Unsorted module map of <module ID, header ID>
///
TRACCC_DEVICE inline void make_module_map(
    std::size_t globalIndex,
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<thrust::pair<geometry_id, unsigned int>>
        module_map_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/finding/device/impl/make_module_map.ipp"
