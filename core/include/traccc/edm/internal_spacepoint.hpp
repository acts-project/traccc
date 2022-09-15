/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include
#include "traccc/edm/spacepoint.hpp"

// detray core
#include <detray/utils/invalid_values.hpp>

// VecMem include(s).
#include <vecmem/containers/data/jagged_vector_buffer.hpp>
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/containers/vector.hpp>

// std
#include <algorithm>

namespace traccc {

/// Item: A internal spacepoint definition
template <typename spacepoint_t>
struct internal_spacepoint {

    // FIXME: geometry_id is hard-coded here
    using spacepoint_container_type = host_container<geometry_id, spacepoint_t>;
    using link_type = typename spacepoint_container_type::link_type;

    link_type m_link;

    scalar m_x;
    scalar m_y;
    scalar m_z;
    scalar m_r;

    internal_spacepoint() = default;

    template <typename spacepoint_container_t>
    TRACCC_HOST_DEVICE internal_spacepoint(
        const spacepoint_container_t& sp_container, const link_type& sp_link,
        const vector2& offsetXY)
        : m_link(sp_link) {
        const spacepoint_t& sp = sp_container.at(sp_link);
        m_x = sp.global[0] - offsetXY[0];
        m_y = sp.global[1] - offsetXY[1];
        m_z = sp.global[2];
        m_r = std::sqrt(m_x * m_x + m_y * m_y);
    }

    TRACCC_HOST_DEVICE
    internal_spacepoint(const link_type& sp_link) : m_link(sp_link) {

        m_x = 0;
        m_y = 0;
        m_z = 0;
        m_r = 0;
    }

    TRACCC_HOST_DEVICE
    static inline internal_spacepoint<spacepoint_t> invalid_value() {

        link_type l = {detray::invalid_value<decltype(l.first)>(),
                       detray::invalid_value<decltype(l.second)>()};

        return internal_spacepoint<spacepoint_t>({std::move(l)});
    }

    TRACCC_HOST_DEVICE const scalar& x() const { return m_x; }

    TRACCC_HOST_DEVICE
    scalar y() const { return m_y; }

    TRACCC_HOST_DEVICE
    scalar z() const { return m_z; }

    TRACCC_HOST_DEVICE
    scalar radius() const { return m_r; }

    TRACCC_HOST_DEVICE
    scalar phi() const { return algebra::math::atan2(m_y, m_x); }

    TRACCC_HOST_DEVICE
    scalar varianceR() const { return 0.; }

    TRACCC_HOST_DEVICE
    scalar varianceZ() const { return 0.; }
};

template <typename spacepoint_t>
inline bool operator<(const internal_spacepoint<spacepoint_t>& lhs,
                      const internal_spacepoint<spacepoint_t>& rhs) {
    return (lhs.radius() < rhs.radius());
}

}  // namespace traccc
