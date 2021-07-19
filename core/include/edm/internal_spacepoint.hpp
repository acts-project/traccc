/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include
#include "edm/spacepoint.hpp"
#include "utils/arch_qualifiers.hpp"

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

struct neighbor_idx {
    size_t counts;
    /// global_indices: the global indices of neighbor bins provided by axis
    /// class size of 9 is from (3 neighbors on z-axis) x (3 neighbors on
    /// phi-axis)
    size_t global_indices[9];
    /// vector_indices: the actual indices of neighbor bins, which are used for
    /// navigating internal_spacepoint_container
    size_t vector_indices[9];
};

/// Header: bin information (global bin index, neighborhood bin indices)
struct bin_information {
    size_t global_index;
    neighbor_idx bottom_idx;
    neighbor_idx top_idx;
};

inline bool operator==(const bin_information& lhs, const bin_information& rhs) {
    return (lhs.global_index == rhs.global_index);
}

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

    __CUDA_HOST_DEVICE__
    internal_spacepoint(const spacepoint& sp, const vector3& globalPos,
                        const vector2& offsetXY, const vector2& variance)
        : m_sp(sp) {
        m_x = globalPos[0] - offsetXY[0];
        m_y = globalPos[1] - offsetXY[1];
        m_z = globalPos[2];
        m_r = std::sqrt(m_x * m_x + m_y * m_y);
        m_varianceR = variance[0];
        m_varianceZ = variance[1];
    }
    __CUDA_HOST_DEVICE__
    internal_spacepoint(const internal_spacepoint<spacepoint>& sp)
        : m_sp(sp.sp()) {
        m_x = sp.m_x;
        m_y = sp.m_y;
        m_z = sp.m_z;
        m_r = sp.m_r;
        m_varianceR = sp.m_varianceR;
        m_varianceZ = sp.m_varianceZ;
    }

    __CUDA_HOST_DEVICE__
    internal_spacepoint& operator=(const internal_spacepoint<spacepoint>& sp) {
        m_x = sp.m_x;
        m_y = sp.m_y;
        m_z = sp.m_z;
        m_r = sp.m_r;
        m_varianceR = sp.m_varianceR;
        m_varianceZ = sp.m_varianceZ;
        return *this;
    }

    __CUDA_HOST_DEVICE__
    const float& x() const { return m_x; }

    __CUDA_HOST_DEVICE__
    const float& y() const { return m_y; }

    __CUDA_HOST_DEVICE__
    const float& z() const { return m_z; }

    __CUDA_HOST_DEVICE__
    const float& radius() const { return m_r; }

    __CUDA_HOST_DEVICE__
    float phi() const { return atan2f(m_y, m_x); }

    __CUDA_HOST_DEVICE__
    const float& varianceR() const { return m_varianceR; }

    __CUDA_HOST_DEVICE__
    const float& varianceZ() const { return m_varianceZ; }

    __CUDA_HOST_DEVICE__
    const spacepoint& sp() const { return m_sp; }
};

/// Container of internal_spacepoint belonging to one spacepoint bin
template <template <typename> class vector_t>
using internal_spacepoint_collection =
    vector_t<internal_spacepoint<spacepoint> >;

/// Convenience declaration for the internal_spacepoint collection type to use
/// in host code
using host_internal_spacepoint_collection =
    internal_spacepoint_collection<vecmem::vector>;

/// Convenience declaration for the internal_spacepoint collection type to use
/// in device code
using device_internal_spacepoint_collection =
    internal_spacepoint_collection<vecmem::device_vector>;

/// Convenience declaration for the internal_spacepoint container type to use in
/// host code
using host_internal_spacepoint_container =
    host_container<bin_information, internal_spacepoint<spacepoint> >;

/// Convenience declaration for the internal_spacepoint container type to use in
/// device code
using device_internal_spacepoint_container =
    device_container<bin_information, internal_spacepoint<spacepoint> >;

/// Convenience declaration for the internal_spacepoint container data type to
/// use in host code
using internal_spacepoint_container_data =
    container_data<bin_information, internal_spacepoint<spacepoint> >;

/// Convenience declaration for the internal_spacepoint container buffer type to
/// use in host code
using internal_spacepoint_container_buffer =
    container_buffer<bin_information, internal_spacepoint<spacepoint> >;

/// Convenience declaration for the internal_spacepoint container view type to
/// use in host code
using internal_spacepoint_container_view =
    container_view<bin_information, internal_spacepoint<spacepoint> >;

inline size_t find_vector_id_from_global_id(
    size_t global_bin, vecmem::vector<bin_information>& headers) {
    auto iterator =
        std::find_if(headers.begin(), headers.end(),
                     [&global_bin](const bin_information& bin_info) {
                         return bin_info.global_index == global_bin;
                     });

    return std::distance(headers.begin(), iterator);
}

inline void fill_vector_id(neighbor_idx& neighbor,
                           vecmem::vector<bin_information>& headers) {
    for (size_t i = 0; i < neighbor.counts; ++i) {
        auto global_id = neighbor.global_indices[i];
        auto vector_id = find_vector_id_from_global_id(global_id, headers);
        assert(vector_id != headers.size());
        neighbor.vector_indices[i] = vector_id;
    }
}

/// Fill vector_indices of header of internal_spacepoint_container
///
inline void fill_vector_id(host_internal_spacepoint_container& isp_container) {
    for (size_t i = 0; i < isp_container.headers.size(); ++i) {
        auto& bot_neighbors = isp_container.headers[i].bottom_idx;
        auto& top_neighbors = isp_container.headers[i].top_idx;

        fill_vector_id(bot_neighbors, isp_container.headers);
        fill_vector_id(top_neighbors, isp_container.headers);
    }
}

}  // namespace traccc
