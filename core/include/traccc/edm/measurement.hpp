/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/container.hpp"

// System include(s).
#include <vector>

namespace traccc {

/// A measurement definition:
/// fix to two-dimensional here
struct measurement {
    point2 local = {0., 0.};
    variance2 variance = {0., 0.};

    TRACCC_HOST_DEVICE
    static inline measurement invalid_value() {
        return measurement({{0., 0.}, {0., 0.}});
    }
};

inline bool operator<(const measurement& lhs, const measurement& rhs) {
    if (lhs.local[0] < rhs.local[0]) {
        return true;
    }
    return false;
}

inline bool operator==(const measurement& lhs, const measurement& rhs) {
    return (std::abs(lhs.local[0] - rhs.local[0]) < float_epsilon &&
            std::abs(lhs.local[1] - rhs.local[1]) < float_epsilon &&
            std::abs(lhs.variance[0] - rhs.variance[0]) < float_epsilon &&
            std::abs(lhs.variance[1] - rhs.variance[1]) < float_epsilon);
}

/// Container of measurements belonging to one detector module
template <template <typename> class vector_t>
using measurement_collection = vector_t<measurement>;

/// Convenience declaration for the measurement collection type to use in host
/// code
using host_measurement_collection = measurement_collection<vecmem::vector>;
/// Convenience declaration for the measurement collection type to use in device
/// code
using device_measurement_collection =
    measurement_collection<vecmem::device_vector>;

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
