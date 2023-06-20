/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Acts include(s).
#include <Acts/Geometry/GeometryHierarchyMap.hpp>
#include <Acts/Utilities/BinUtility.hpp>

namespace traccc {

/// Type describing the digitization configuration of a detector module
struct module_digitization_config {
    Acts::BinUtility segmentation;
};

/// Type describing the digitization configuration for the whole detector
using digitization_config =
    Acts::GeometryHierarchyMap<module_digitization_config>;

}  // namespace traccc
