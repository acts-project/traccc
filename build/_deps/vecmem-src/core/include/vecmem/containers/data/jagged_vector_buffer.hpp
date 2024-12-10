/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/buffer_type.hpp"
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/memory/unique_ptr.hpp"

// System include(s).
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

namespace vecmem {
namespace data {

/// Object owning all the data of a jagged vector
///
/// This type is needed for the explicit memory management of jagged vectors.
///
template <typename TYPE>
class jagged_vector_buffer : public jagged_vector_view<TYPE> {

public:
    /// The base type used by this class
    typedef jagged_vector_view<TYPE> base_type;
    /// Use the base class's @c size_type
    typedef typename base_type::size_type size_type;
    /// Use the base class's @c value_type
    typedef typename base_type::value_type value_type;
    /// Pointer type to the jagged array
    typedef typename base_type::pointer pointer;

    /// @name Checks on the type of the array element
    /// @{

    /// Make sure that the template type does not have a custom destructor
    static_assert(std::is_trivially_destructible<TYPE>::value,
                  "vecmem::data::jagged_vector_buffer can not handle types "
                  "with custom destructors");
    /// Make sure that @c vecmem::data::vector_view does not have a custom
    /// destructor
    static_assert(std::is_trivially_destructible<value_type>::value,
                  "vecmem::data::jagged_vector_buffer can not handle types "
                  "with custom destructors");

    /// @}

    /// Default constructor
    jagged_vector_buffer();

    /// Constructor from an existing @c vecmem::data::jagged_vector_view
    ///
    /// @param other The existing @c vecmem::data::jagged_vector_view object
    ///        that this buffer should mirror.
    /// @param resource The device accessible memory resource, which may also
    ///        be host accessible.
    /// @param host_access_resource An optional host accessible memory
    ///        resource. Needed if @c resource is not host accessible.
    /// @param type The type (resizable or not) of the buffer
    ///
    template <typename OTHERTYPE,
              std::enable_if_t<std::is_convertible<TYPE, OTHERTYPE>::value,
                               bool> = true>
    jagged_vector_buffer(const jagged_vector_view<OTHERTYPE>& other,
                         memory_resource& resource,
                         memory_resource* host_access_resource = nullptr,
                         buffer_type type = buffer_type::fixed_size);

    /// Constructor from a vector of ("inner vector") sizes
    ///
    /// @param capacities Simple vector holding the capacities/sizes of the
    ///        "inner vectors" for the jagged vector buffer.
    /// @param resource The device accessible memory resource, which may also
    ///        be host accessible.
    /// @param host_access_resource An optional host accessible memory
    ///        resource. Needed if @c resource is not host accessible.
    /// @param type The type (resizable or not) of the buffer
    ///
    template <typename SIZE_TYPE = std::size_t,
              std::enable_if_t<std::is_integral<SIZE_TYPE>::value &&
                                   std::is_unsigned<SIZE_TYPE>::value,
                               bool> = true>
    jagged_vector_buffer(const std::vector<SIZE_TYPE>& capacities,
                         memory_resource& resource,
                         memory_resource* host_access_resource = nullptr,
                         buffer_type type = buffer_type::fixed_size);

    /// Move constructor
    jagged_vector_buffer(jagged_vector_buffer&&) = default;

    /// Move assignment
    jagged_vector_buffer& operator=(jagged_vector_buffer&&) = default;

private:
    /// Data object for the @c vecmem::data::vector_view array
    vecmem::unique_alloc_ptr<value_type[]> m_outer_memory;
    /// Data object for the @c vecmem::data::vector_view array on the host
    vecmem::unique_alloc_ptr<value_type[]> m_outer_host_memory;
    /// Data object owning the memory of the "inner vectors"
    vecmem::unique_alloc_ptr<char[]> m_inner_memory;

};  // class jagged_vector_buffer

}  // namespace data

/// Helper function creating a @c vecmem::data::jagged_vector_view object
template <typename TYPE>
data::jagged_vector_view<TYPE>& get_data(
    data::jagged_vector_buffer<TYPE>& data);

/// Helper function creating a @c vecmem::data::jagged_vector_view object
template <typename TYPE>
const data::jagged_vector_view<TYPE>& get_data(
    const data::jagged_vector_buffer<TYPE>& data);

}  // namespace vecmem

// Include the implementation.
#include "vecmem/containers/impl/jagged_vector_buffer.ipp"
