/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include
#include "edm/spacepoint.hpp"

// VecMem include(s).
#include <vecmem/containers/data/jagged_vector_buffer.hpp>
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/containers/vector.hpp>

// std
#include <algorithm>

// detray core
#include <detray/definitions/invalid_values.hpp>

namespace traccc {

/// Item: A internal spacepoint definition
template <typename spacepoint>
struct internal_spacepoint {
    scalar m_x;
    scalar m_y;
    scalar m_z;
    scalar m_r;
    scalar m_varianceR;
    scalar m_varianceZ;
    spacepoint m_sp;

    internal_spacepoint() = default;

    TRACCC_HOST_DEVICE
    internal_spacepoint(const spacepoint& sp, const vector2& offsetXY)
        : m_sp(sp) {
        m_x = sp.global[0] - offsetXY[0];
        m_y = sp.global[1] - offsetXY[1];
        m_z = sp.global[2];
        m_r = std::sqrt(m_x * m_x + m_y * m_y);

        // Need to fix this part
        m_varianceR = sp.variance[0];
        m_varianceZ = sp.variance[1];
    }

    TRACCC_HOST_DEVICE
    internal_spacepoint(const internal_spacepoint<spacepoint>& sp)
        : m_sp(sp.sp()) {
        m_x = sp.m_x;
        m_y = sp.m_y;
        m_z = sp.m_z;
        m_r = sp.m_r;
        m_varianceR = sp.m_varianceR;
        m_varianceZ = sp.m_varianceZ;
    }

    TRACCC_HOST_DEVICE
    internal_spacepoint& operator=(internal_spacepoint&& sp) {
        m_sp = std::move(sp.sp());
        m_x = std::move(sp.m_x);
        m_y = std::move(sp.m_y);
        m_z = std::move(sp.m_z);
        m_r = std::move(sp.m_r);
        m_varianceR = std::move(sp.m_varianceR);
        m_varianceZ = std::move(sp.m_varianceZ);
        return *this;
    }

    TRACCC_HOST_DEVICE
    internal_spacepoint& operator=(const internal_spacepoint<spacepoint>& sp) {
        m_sp = sp.sp();
        m_x = sp.m_x;
        m_y = sp.m_y;
        m_z = sp.m_z;
        m_r = sp.m_r;
        m_varianceR = sp.m_varianceR;
        m_varianceZ = sp.m_varianceZ;
        return *this;
    }

    TRACCC_HOST_DEVICE
    static inline internal_spacepoint<spacepoint> invalid_value() {
        spacepoint sp = spacepoint::invalid_value();
        return internal_spacepoint<spacepoint>({sp, {0., 0.}});
    }

    TRACCC_HOST_DEVICE
    const scalar& x() const { return m_x; }

    TRACCC_HOST_DEVICE
    const scalar& y() const { return m_y; }

    TRACCC_HOST_DEVICE
    const scalar& z() const { return m_z; }

    TRACCC_HOST_DEVICE
    const scalar& radius() const { return m_r; }

    TRACCC_HOST_DEVICE
    scalar phi() const { return atan2f(m_y, m_x); }

    TRACCC_HOST_DEVICE
    const scalar& varianceR() const { return m_varianceR; }

    TRACCC_HOST_DEVICE
    const scalar& varianceZ() const { return m_varianceZ; }

    TRACCC_HOST_DEVICE
    const spacepoint& sp() const { return m_sp; }
};

template <typename spacepoint_t>
inline bool operator<(const internal_spacepoint<spacepoint_t>& lhs,
                      const internal_spacepoint<spacepoint_t>& rhs) {
    return (lhs.radius() < rhs.radius());
}

}  // namespace traccc
