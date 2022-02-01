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
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/edm/measurement.hpp"

// System include(s).
#include <vector>

namespace traccc {

/// A spacepoint definition: global position and errors
struct spacepoint {
    point3 global = {0., 0., 0.};
    variance3 variance = {0., 0., 0.};
    measurement meas;

    TRACCC_HOST_DEVICE
    static inline spacepoint invalid_value() {
        measurement ms = measurement::invalid_value();
        return spacepoint({{0., 0., 0.}, {0., 0., 0.}, ms});
    }

    TRACCC_HOST_DEVICE
    const scalar& x() const { return global[0]; }
    TRACCC_HOST_DEVICE
    const scalar& y() const { return global[1]; }
    TRACCC_HOST_DEVICE
    const scalar& z() const { return global[2]; }
    TRACCC_HOST_DEVICE
    scalar radius() const {
        return std::sqrt(global[0] * global[0] + global[1] * global[1]);
    }
};

inline bool operator==(const spacepoint& lhs, const spacepoint& rhs) {
    if (std::abs(lhs.global[0] - rhs.global[0]) < float_epsilon &&
        std::abs(lhs.global[1] - rhs.global[1]) < float_epsilon &&
        std::abs(lhs.global[2] - rhs.global[2]) < float_epsilon) {
        return true;
    }
    return false;
}

/// Container of spacepoints belonging to one detector module
template <template <typename> class vector_t>
using spacepoint_collection = vector_t<spacepoint>;

/// Convenience declaration for the spacepoint collection type to use in host
/// code
using host_spacepoint_collection = spacepoint_collection<vecmem::vector>;

/// Convenience declaration for the spacepoint collection type to use in device
/// code
using device_spacepoint_collection =
    spacepoint_collection<vecmem::device_vector>;

/// Convenience declaration for the spacepoint container type to use in host
/// code
using host_spacepoint_container = host_container<geometry_id, spacepoint>;

/// Convenience declaration for the spacepoint container type to use in device
/// code
using device_spacepoint_container = device_container<geometry_id, spacepoint>;

/// Convenience declaration for the spacepoint container data type to use in
/// host code
using spacepoint_container_data = container_data<geometry_id, spacepoint>;

/// Convenience declaration for the spacepoint container buffer type to use in
/// host code
using spacepoint_container_buffer = container_buffer<geometry_id, spacepoint>;

/// Convenience declaration for the spacepoint container view type to use in
/// host code
using spacepoint_container_view = container_view<geometry_id, spacepoint>;

}  // namespace traccc
