/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/device.hpp"
#include "vecmem/edm/schema.hpp"
#include "vecmem/edm/view.hpp"

#if __cplusplus >= 201700L
#include "vecmem/edm/buffer.hpp"
#include "vecmem/edm/data.hpp"
#include "vecmem/edm/host.hpp"
#endif  // __cplusplus >= 201700L

/// Namespace for the types serving as base classes for an SoA event data model
namespace vecmem {
namespace edm {

/// Description of a container, with all of the types needed to use it
///
/// @tparam INTERFACE Class providing a convenient user interface to the
///                   container
/// @tparam ...VARIABLES The variable types stored in the container
///
template <template <typename> class INTERFACE, typename... VARIABLES>
struct container {

    /// Interface to the container
    template <typename DERIVED>
    using interface_type = INTERFACE<DERIVED>;
    /// Schema for this container
    using schema_type = schema<VARIABLES...>;
    /// Constant version of the schema
    using const_schema_type = typename details::add_const<schema_type>::type;

#if __cplusplus >= 201700L
    /// Host container type
    using host = interface_type<vecmem::edm::host<schema_type, interface_type>>;

    /// (Non-const) Data type
    using data = vecmem::edm::data<schema_type>;
    /// (Const) Data type
    using const_data = vecmem::edm::data<const_schema_type>;

    /// Buffer type
    using buffer = vecmem::edm::buffer<schema_type>;
#endif  // __cplusplus >= 201700L

    /// (Non-const) Device container type
    using device =
        interface_type<vecmem::edm::device<schema_type, interface_type>>;
    /// (Const) Device container type
    using const_device =
        interface_type<vecmem::edm::device<const_schema_type, interface_type>>;

    /// (Non-const) View type
    using view = vecmem::edm::view<schema_type>;
    /// (Const) view type
    using const_view = vecmem::edm::view<const_schema_type>;

};  // struct container

}  // namespace edm
}  // namespace vecmem
