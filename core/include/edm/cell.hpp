/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include(s).
#include "definitions/algebra.hpp"
#include "definitions/primitives.hpp"

// VecMem include(s).
#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/containers/vector.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/data/jagged_vector_buffer.hpp>
#include <vecmem/containers/data/vector_buffer.hpp>

// System include(s).
#include <limits>

namespace traccc {

    /// @name Types to use in algorithmic code
    /// @{

    /// Channel identifier type
    using channel_id = unsigned int;

    /// A cell definition:
    ///
    /// maximum two channel identifiers
    /// and one activiation value, such as a time stamp
    struct cell {
        channel_id channel0 = 0;
        channel_id channel1 = 0;
        scalar activation = 0.;
        scalar time = 0.;
    };

    /// Container of cells belonging to one detector module
    template< template< typename > class vector_t >
    using cell_collection = vector_t< cell >;

    /// Convenience declaration for the cell collection type to use in host code
    using host_cell_collection = cell_collection< vecmem::vector >;
    /// Convenience declaration for the cell collection type to use in device code
    using device_cell_collection = cell_collection< vecmem::device_vector >;

    /// Header information for all of the cells in a specific detector module
    ///
    /// It is handled separately from the list of all of the cells belonging to
    /// the detector module, to be able to lay out the data in memory in a way
    /// that is more friendly towards accelerators.
    ///
    struct cell_module {

        event_id event = 0;
        geometry_id module = 0;
        transform3 placement = transform3{};

        channel_id range0[ 2 ] = { std::numeric_limits< channel_id >::max(), 0 };
        channel_id range1[ 2 ] = { std::numeric_limits< channel_id >::max(), 0 };

    }; // struct cell_module

    /// Container describing all of the cells in a given event
    ///
    /// This is the "main" cell container of the code, holding all relevant
    /// information about all of the cells in a given event.
    ///
    /// It can be instantiated with different vector types, to be able to use
    /// the same container type in both host and device code.
    ///
    template< template< typename > class vector_t,
              template< typename > class jagged_vector_t >
    class cell_container {

    public:
        /// @name Type definitions
        /// @{

        /// Vector type used by the cell container
        template< typename T >
        using vector_type = vector_t< T >;
        /// Jagged vector type used by the cell container
        template< typename T >
        using jagged_vector_type = jagged_vector_t< T >;

        /// The cell module vector type
        using cell_module_vector = vector_type< cell_module >;
        /// The cell vector type
        using cell_vector = jagged_vector_type< cell >;

        /// @}

        /// Headers for all of the modules (holding cells) in the event
        cell_module_vector modules;
        /// All of the cells in the event
        cell_vector cells;

    }; // class cell_container

    /// Convenience declaration for the cell container type to use in host code
    using host_cell_container =
        cell_container< vecmem::vector, vecmem::jagged_vector >;
    /// Convenience declaration for the cell container type to use in device code
    using device_cell_container =
        cell_container< vecmem::device_vector, vecmem::jagged_device_vector >;

    /// @}

    /// @name Types used to send data back and forth between host and device code
    /// @{

    /// Structure holding (some of the) data about the cells in host code
    struct cell_container_data {
        vecmem::data::vector_view< cell_module > modules;
        vecmem::data::jagged_vector_data< cell > cells;
    }; // struct cell_container_data

    /// Structure holding (all of the) data about the cells in host code
    struct cell_container_buffer {
        vecmem::data::vector_buffer< cell_module > modules;
        vecmem::data::jagged_vector_buffer< cell > cells;
    }; // struct cell_container_data

    /// Structure used to send the data about the cells to device code
    ///
    /// This is the type that can be passed to device code as-is. But since in
    /// host code one needs to manage the data describing a
    /// @c traccc::cell_container either using @c traccc::cell_container_data or
    /// @c traccc::cell_container_buffer, it needs to have constructors from
    /// both of those types.
    ///
    /// In fact it needs to be created from one of those types, as such an
    /// object can only function if an instance of one of those types exists
    /// alongside it as well.
    ///
    struct cell_container_view {

        /// Constructor from a @c cell_container_data object
        cell_container_view( const cell_container_data& data )
        : modules( data.modules ), cells( data.cells ) {}

        /// Constructor from a @c cell_container_buffer object
        cell_container_view( const cell_container_buffer& buffer )
        : modules( buffer.modules ), cells( buffer.cells ) {}

        /// View of the data describing the headers of the cell holding modules
        vecmem::data::vector_view< cell_module > modules;
        /// View of the data describing all of the cells
        vecmem::data::jagged_vector_view< cell > cells;

    }; // struct cell_container_view

    /// Helper function for making a "simple" object out of the cell container
    cell_container_data get_data( host_cell_container& cc ) {
        return { { vecmem::get_data( cc.modules ) },
                 { vecmem::get_data( cc.cells ) } };
    }

    /// @}

} // namespace traccc
