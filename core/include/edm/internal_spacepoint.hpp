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
#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/containers/vector.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/data/jagged_vector_buffer.hpp>
#include <vecmem/containers/data/vector_buffer.hpp>


namespace traccc {

    /// Header: bin information (global bin index, neighborhood bin indices)
    struct bin_info{
	size_t global_index;
	size_t num_bottom_bin_indices;
	size_t num_top_bin_indices;
	std::array< size_t, size_t(9u) > bottom_bin_indices;
	std::array< size_t, size_t(9u) > top_bin_indices;
    };
    bool operator==(const bin_info& lhs, const bin_info& rhs){
	return (lhs.global_index == rhs.global_index);
    }
    
    
    /// Item: A internal spacepoint definition
    template< typename spacepoint >
    struct internal_spacepoint{
	scalar m_x;
	scalar m_y;
	scalar m_z;
	scalar m_r;
	scalar m_varianceR;
	scalar m_varianceZ;
	const spacepoint& m_sp;

	internal_spacepoint(const spacepoint& sp, const vector3& globalPos,
			    const vector2& offsetXY,
			    const vector2& variance): m_sp(sp) {	  
	    m_x = globalPos[0] - offsetXY[0];
	    m_y = globalPos[1] - offsetXY[1];
	    m_z = globalPos[2];
	    m_r = std::sqrt(m_x * m_x + m_y * m_y);
	    m_varianceR = variance[0];
	    m_varianceZ = variance[1];	    
	}
	internal_spacepoint(const internal_spacepoint<spacepoint>& sp)
	    : m_sp(sp.sp()){
	    m_x = sp.m_x;
	    m_y = sp.m_y;
	    m_z = sp.m_z;
	    m_r = sp.m_r;
	    m_varianceR = sp.m_varianceR;
	    m_varianceZ = sp.m_varianceZ;
	}

	const float& x() const { return m_x; }
	const float& y() const { return m_y; }
	const float& z() const { return m_z; }
	const float& radius() const { return m_r; }
	float phi() const { return atan2f(m_y, m_x); }
	const float& varianceR() const { return m_varianceR; }
	const float& varianceZ() const { return m_varianceZ; }
	const spacepoint& sp() const { return m_sp; }
    };
    
    /// Container of internal_spacepoint belonging to one detector module
    template< template< typename > class vector_t >
    using internal_spacepoint_collection = vector_t< internal_spacepoint<spacepoint> >;

    /// Convenience declaration for the internal_spacepoint collection type to use in host code
    using host_internal_spacepoint_collection
    = internal_spacepoint_collection< vecmem::vector >;

    /// Convenience declaration for the internal_spacepoint collection type to use in device code
    using device_internal_spacepoint_collection
    = internal_spacepoint_collection< vecmem::device_vector >;
    
    /// Convenience declaration for the internal_spacepoint container type to use in host code
    /// header: global bin index / neighborhood index
    /// item  : internal spacepoint
    using host_internal_spacepoint_container
    = host_container< bin_info, internal_spacepoint<spacepoint> >;

    /// Convenience declaration for the internal_spacepoint container type to use in device code
    using device_internal_spacepoint_container
    = device_container< bin_info, internal_spacepoint<spacepoint> >;

    /// Convenience declaration for the internal_spacepoint container data type to use in host code
    using internal_spacepoint_container_data
    = container_data< bin_info, internal_spacepoint<spacepoint> >;

    /// Convenience declaration for the internal_spacepoint container buffer type to use in host code
    using internal_spacepoint_container_buffer
    = container_buffer< bin_info, internal_spacepoint<spacepoint> >;

    /// Convenience declaration for the internal_spacepoint container view type to use in host code
    using internal_spacepoint_container_view
    = container_view< bin_info, internal_spacepoint<spacepoint> >;

    
} // namespace traccc
