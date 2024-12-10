/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// VecMem include(s).
#include "vecmem/containers/data/jagged_vector_buffer.hpp"
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "vecmem/containers/data/vector_buffer.hpp"
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/edm/details/schema_traits.hpp"
#include "vecmem/edm/host.hpp"
#include "vecmem/edm/view.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/utils/abstract_event.hpp"
#include "vecmem/utils/attributes.hpp"
#include "vecmem/vecmem_core_export.hpp"

// System include(s).
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

namespace vecmem {

/// Class implementing (synchronous) host <-> device memory copies
///
/// Since most of the logic of explicitly copying the payload of vecmem
/// containers between the host and device is independent of the exact GPU
/// language used, this common base class is used for implementing most of
/// that logic.
///
/// Language specific @c copy classes should only need to re-implement the
/// @c do_copy function, everything else should be provided by this class.
///
class VECMEM_CORE_EXPORT copy {

public:
    /// Wrapper struct around the @c copy_type enumeration
    ///
    /// The code does not use "enum struct type" to declare the copy type, as
    /// that unnecessarily makes it hard to use these values as array indices
    /// in client code.
    ///
    struct type {
        /// Types of memory copies to handle
        enum copy_type {
            /// Copy operation between the host and a device
            host_to_device = 0,
            /// Copy operation between a device and the host
            device_to_host = 1,
            /// Copy operation on the host
            host_to_host = 2,
            /// Copy operation between two devices
            device_to_device = 3,
            /// Unknown copy type, determined at runtime
            unknown = 4,
            /// The number of copy types, useful for technical reasons
            count = 5
        };  // enum copy_type
    };      // struct type

    /// Event type used by the copy class
    using event_type = std::unique_ptr<abstract_event>;

    /// @name 1-dimensional vector data handling functions
    /// @{

    /// Virtual destructor
    virtual ~copy() {}

    /// Set up the internal state of a vector buffer correctly on a device
    template <typename TYPE>
    VECMEM_NODISCARD event_type setup(data::vector_view<TYPE> data) const;

    /// Set all bytes of the vector to some value
    template <typename TYPE>
    VECMEM_NODISCARD event_type memset(data::vector_view<TYPE> data,
                                       int value) const;

    /// Copy a 1-dimensional vector to the specified memory resource
    template <typename TYPE>
    data::vector_buffer<std::remove_cv_t<TYPE>> to(
        const data::vector_view<TYPE>& data, memory_resource& resource,
        type::copy_type cptype = type::unknown) const;

    /// Copy a 1-dimensional vector's data between two existing memory blocks
    template <typename TYPE>
    VECMEM_NODISCARD event_type
    operator()(const data::vector_view<std::add_const_t<TYPE>>& from,
               data::vector_view<TYPE> to,
               type::copy_type cptype = type::unknown) const;

    /// Copy a 1-dimensional vector's data into a vector object
    template <typename TYPE, typename ALLOC>
    VECMEM_NODISCARD event_type
    operator()(const data::vector_view<std::add_const_t<TYPE>>& from,
               std::vector<TYPE, ALLOC>& to,
               type::copy_type cptype = type::unknown) const;

    /// Helper function for getting the size of a resizable 1D buffer
    template <typename TYPE>
    typename data::vector_view<TYPE>::size_type get_size(
        const data::vector_view<TYPE>& data) const;

    /// @}

    /// @name Jagged vector data handling functions
    /// @{

    /// Copy the internal state of a jagged vector buffer to the target device
    template <typename TYPE>
    VECMEM_NODISCARD event_type
    setup(data::jagged_vector_view<TYPE> data) const;

    /// Set all bytes of the jagged vector to some value
    template <typename TYPE>
    VECMEM_NODISCARD event_type memset(data::jagged_vector_view<TYPE> data,
                                       int value) const;

    /// Copy a jagged vector to the specified memory resource
    template <typename TYPE>
    data::jagged_vector_buffer<std::remove_cv_t<TYPE>> to(
        const data::jagged_vector_view<TYPE>& data, memory_resource& resource,
        memory_resource* host_access_resource = nullptr,
        type::copy_type cptype = type::unknown) const;

    /// Copy a jagged vector's data between two existing allocations
    template <typename TYPE>
    VECMEM_NODISCARD event_type
    operator()(const data::jagged_vector_view<std::add_const_t<TYPE>>& from,
               data::jagged_vector_view<TYPE> to,
               type::copy_type cptype = type::unknown) const;

    /// Copy a jagged vector's data into a vector object
    template <typename TYPE, typename ALLOC1, typename ALLOC2>
    VECMEM_NODISCARD event_type
    operator()(const data::jagged_vector_view<std::add_const_t<TYPE>>& from,
               std::vector<std::vector<TYPE, ALLOC2>, ALLOC1>& to,
               type::copy_type cptype = type::unknown) const;

    /// Helper function for getting the sizes of a resizable jagged vector
    template <typename TYPE>
    std::vector<typename data::vector_view<TYPE>::size_type> get_sizes(
        const data::jagged_vector_view<TYPE>& data) const;

    /// Helper function for setting the sizes of a resizable jagged vector
    template <typename TYPE>
    VECMEM_NODISCARD event_type set_sizes(
        const std::vector<typename data::vector_view<TYPE>::size_type>& sizes,
        data::jagged_vector_view<TYPE> data) const;

    /// @}

    /// @name SoA container handling functions
    /// @{

    /// Set up the internal state of a buffer correctly on a device
    template <typename SCHEMA>
    VECMEM_NODISCARD event_type setup(edm::view<SCHEMA> data) const;

    /// Set all bytes of the container to some value
    template <typename... VARTYPES>
    VECMEM_NODISCARD event_type memset(edm::view<edm::schema<VARTYPES...>> data,
                                       int value) const;

    /// Copy between two views
    template <typename... VARTYPES>
    VECMEM_NODISCARD event_type operator()(
        const edm::view<edm::details::add_const_t<edm::schema<VARTYPES...>>>&
            from,
        edm::view<edm::schema<VARTYPES...>> to,
        type::copy_type cptype = type::unknown) const;

    /// Copy from a view, into a host container
    template <typename... VARTYPES, template <typename> class INTERFACE>
    VECMEM_NODISCARD event_type operator()(
        const edm::view<edm::details::add_const_t<edm::schema<VARTYPES...>>>&
            from,
        edm::host<edm::schema<VARTYPES...>, INTERFACE>& to,
        type::copy_type cptype = type::unknown) const;

    /// Get the (outer) size of a resizable SoA container
    template <typename... VARTYPES>
    typename edm::view<edm::schema<VARTYPES...>>::size_type get_size(
        const edm::view<edm::schema<VARTYPES...>>& data) const;

    /// @}

protected:
    /// Perform a "low level" memory copy
    virtual void do_copy(std::size_t size, const void* from, void* to,
                         type::copy_type cptype) const;
    /// Perform a "low level" memory filling operation
    virtual void do_memset(std::size_t size, void* ptr, int value) const;
    /// Create an event for synchronization
    VECMEM_NODISCARD virtual event_type create_event() const;

private:
    /// Implementation for the 1D vector copy operator
    template <typename TYPE>
    bool copy_view_impl(const data::vector_view<std::add_const_t<TYPE>>& from,
                        data::vector_view<TYPE> to,
                        type::copy_type cptype) const;
    /// Implementation of the jagged vector copy operator
    template <typename TYPE>
    bool copy_view_impl(
        const data::jagged_vector_view<std::add_const_t<TYPE>>& from,
        data::jagged_vector_view<TYPE> to, type::copy_type cptype) const;
    /// Helper function performing the copy of a jagged array/vector
    template <typename TYPE>
    void copy_views_impl(
        const std::vector<typename data::vector_view<TYPE>::size_type>& sizes,
        const data::vector_view<std::add_const_t<TYPE>>* from,
        data::vector_view<TYPE>* to, type::copy_type cptype) const;
    /// Helper function performing the copy of a jagged array/vector
    template <typename TYPE>
    void copy_views_contiguous_impl(
        const std::vector<typename data::vector_view<TYPE>::size_type>& sizes,
        const data::vector_view<std::add_const_t<TYPE>>* from,
        data::vector_view<TYPE>* to, type::copy_type cptype) const;
    /// Helper function for getting the sizes of a jagged vector/buffer
    template <typename TYPE>
    std::vector<typename data::vector_view<TYPE>::size_type> get_sizes_impl(
        const data::vector_view<TYPE>* data, std::size_t size) const;
    /// Check if a vector of views occupy a contiguous block of memory
    template <typename TYPE>
    static bool is_contiguous(const data::vector_view<TYPE>* data,
                              std::size_t size);
    /// Implementation for the variadic @c memset function
    template <std::size_t INDEX, typename... VARTYPES>
    void memset_impl(edm::view<edm::schema<VARTYPES...>> data, int value) const;
    /// Implementation for setting the sizes of an SoA container
    template <std::size_t INDEX, typename... VARTYPES,
              template <typename> class INTERFACE>
    void resize_impl(
        const edm::view<edm::details::add_const_t<edm::schema<VARTYPES...>>>&
            from,
        edm::host<edm::schema<VARTYPES...>, INTERFACE>& to,
        type::copy_type cptype) const;
    /// Implementation for the variadic @c copy function (for the sizes)
    template <std::size_t INDEX, typename... VARTYPES>
    void copy_sizes_impl(
        const edm::view<edm::details::add_const_t<edm::schema<VARTYPES...>>>&
            from,
        edm::view<edm::schema<VARTYPES...>> to, type::copy_type cptype) const;
    /// Implementation for the variadic @c copy function (for the payload)
    template <std::size_t INDEX, typename... VARTYPES>
    void copy_payload_impl(
        const edm::view<edm::details::add_const_t<edm::schema<VARTYPES...>>>&
            from,
        edm::view<edm::schema<VARTYPES...>> to, type::copy_type cptype) const;

};  // class copy

}  // namespace vecmem

// Include the implementation.
#include "vecmem/utils/impl/copy.ipp"
