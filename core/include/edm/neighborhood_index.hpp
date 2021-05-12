/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/containers/vector.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/data/jagged_vector_buffer.hpp>
#include <vecmem/containers/data/vector_buffer.hpp>

namespace traccc {

    // neighborhood_index for bin finding
    // header < size_t >: number of neighbor bins
    // item   < size_t >: index of neighbor bins
    
    /// Container of neighborhood_index belonging to one detector module
    template< template< typename > class vector_t >
    using neighborhood_index_collection = vector_t< size_t >;

    /// Convenience declaration for the neighborhood_index collection type to use in host code
    using host_neighborhood_index_collection
    = neighborhood_index_collection< vecmem::vector >;

    /// Convenience declaration for the neighborhood_index collection type to use in device code
    using device_neighborhood_index_collection
    = neighborhood_index_collection< vecmem::device_vector >;

    /// Convenience declaration for the neighborhood_index container type to use in host code
    using host_neighborhood_index_container
    = host_container< size_t, size_t >;

    /// Convenience declaration for the neighborhood_index container type to use in device code
    using device_neighborhood_index_container
    = device_container< size_t, size_t >;

    /// Convenience declaration for the neighborhood_index container data type to use in host code
    using neighborhood_index_container_data
    = container_data< size_t, size_t >;

    /// Convenience declaration for the neighborhood_index container buffer type to use in host code
    using neighborhood_index_container_buffer
    = container_buffer< size_t, size_t >;

    /// Convenience declaration for the neighborhood_index container view type to use in host code
    using neighborhood_index_container_view
    = container_view< size_t, size_t >;
        
} // namespace traccc
