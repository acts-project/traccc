/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <array>
#include <cstdint>

namespace traccc::cuda {
/**
 * @brief Range in three dimensions, defined by minima and maxima in the phi,
 * r, and z dimensions.
 */
struct range3d {
    float phi_min, phi_max, r_min, r_max, z_min, z_max;

    __host__ __device__ __forceinline__ static range3d Infinite() {
        range3d r;

        r.phi_min = std::numeric_limits<float>::lowest();
        r.phi_max = std::numeric_limits<float>::max();
        r.r_min = 0.f;
        r.r_max = std::numeric_limits<float>::max();
        r.z_min = std::numeric_limits<float>::lowest();
        r.z_max = std::numeric_limits<float>::max();

        return r;
    }

    __host__ __device__ __forceinline__ static range3d Degenerate() {
        range3d r;

        r.phi_min = std::numeric_limits<float>::max();
        r.phi_max = std::numeric_limits<float>::lowest();
        r.r_min = std::numeric_limits<float>::max();
        r.r_max = 0.f;
        r.z_min = std::numeric_limits<float>::max();
        r.z_max = std::numeric_limits<float>::lowest();

        return r;
    }

    __host__ __device__ __forceinline__ static range3d Union(const range3d& a,
                                                             const range3d& b) {
        return {std::min(a.phi_min, b.phi_min), std::max(a.phi_max, b.phi_max),
                std::min(a.r_min, b.r_min),     std::max(a.r_max, b.r_max),
                std::min(a.z_min, b.z_min),     std::max(a.z_max, b.z_max)};
    }

    __host__ __device__ __forceinline__ bool intersects(
        const range3d& o) const {
        return phi_min <= o.phi_max && o.phi_min < phi_max &&
               r_min <= o.r_max && o.r_min < r_max && z_min <= o.z_max &&
               o.z_min < z_max;
    }

    __host__ __device__ __forceinline__ bool dominates(const range3d& o) const {
        return phi_min <= o.phi_min && o.phi_max <= phi_max &&
               r_min <= o.r_min && o.r_max <= r_max && z_min <= o.z_min &&
               o.z_max <= z_max;
    }

    __host__ __device__ __forceinline__ bool contains(float phi, float r,
                                                      float z) const {
        return phi_min <= phi && phi <= phi_max && r_min <= r && r <= r_max &&
               z_min <= z && z <= z_max;
    }
};
}  // namespace traccc::cuda
