/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <definitions/primitives.hpp>
#include <utils/arch_qualifiers.hpp>
#include <vector>

#include "container.hpp"

namespace traccc {

/// A spacepoint definition: global position and errors
struct spacepoint {
    point3 global = {0., 0., 0.};
    variance3 variance = {0., 0., 0.};

    __CUDA_HOST_DEVICE__
    const scalar& x() const { return global[0]; }
    __CUDA_HOST_DEVICE__
    const scalar& y() const { return global[1]; }
    __CUDA_HOST_DEVICE__
    const scalar& z() const { return global[2]; }
    __CUDA_HOST_DEVICE__
    scalar radius() const {
        return std::sqrt(global[0] * global[0] + global[1] * global[1]);
    }
};

inline bool operator==(const spacepoint& lhs, const spacepoint& rhs) {
    if (std::abs(lhs.global[0] - rhs.global[0]) < 1e-6 &&
        std::abs(lhs.global[1] - rhs.global[1]) < 1e-6 &&
        std::abs(lhs.global[2] - rhs.global[2]) < 1e-6) {
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
