/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <vector>

#include "collection.hpp"
#include "container.hpp"
#include "definitions/primitives.hpp"

namespace traccc {

/// A measurement definition:
/// fix to two-dimensional here
struct measurement {
    point2 local = {0., 0.};
    variance2 variance = {0., 0.};
};

/// Convenience declaration for the measurement collection type to use in host
/// code
using host_measurement_collection = host_collection<measurement>;

/// Convenience declaration for the measurement collection type to use in device
/// code
using device_measurement_collection = device_collection<measurement>;

/// Convenience declaration for the measurement container type to use in host
/// code
using host_measurement_container = host_container<cell_module, measurement>;

/// Convenience declaration for the measurement container type to use in device
/// code
using device_measurement_container = device_container<cell_module, measurement>;
/// Convenience declaration for the measurement container data type to use in
/// host code
using measurement_container_data = container_data<cell_module, measurement>;

/// Convenience declaration for the measurement container buffer type to use in
/// host code
using measurement_container_buffer = container_buffer<cell_module, measurement>;

/// Convenience declaration for the measurement container view type to use in
/// host code
using measurement_container_view = container_view<cell_module, measurement>;

}  // namespace traccc
