/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/buffer_type.hpp"
#include "vecmem/edm/details/schema_traits.hpp"
#include "vecmem/edm/schema.hpp"
#include "vecmem/edm/view.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/memory/unique_ptr.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <type_traits>
#include <utility>
#include <vector>

namespace vecmem {
namespace edm {

/// Technical base type for @c buffer<schema<VARTYPES...>>
template <typename T>
class buffer;

/// Buffer for a Struct-of-Arrays container
///
/// This type can be used to hold the memory of an entire SoA container in an
/// efficient way. Allowing for block copies of the data between the host and
/// a device.
///
/// The buffer holds on to either 1 or 2 blocks of memory.
///   - Only one block is needed if:
///      * The container doesn't have any jagged vector variables;
///      * The main memory resource used by the buffer is host accessible.
///   - If the container has at least one jagged vector variable and the main
///     memory resource is not host accessible, then the buffer will hold on to
///     two blocks of memory. One allocated on the device and one on the host.
///
/// The "main/device memory block" is structured as follows:
///
///   <table>
///     <caption id="buffer_layout">
///        `vecmem::edm::buffer` memory layout
///     </caption>
///     <tr>
///       <th colspan="3">
///         Fixed size buffer
///       </th>
///     </tr>
///     <tr><th>Bytes</th><th>Description</th><th>Notes</th></tr>
///     <tr>
///       <td>
///         `sizeof(vecmem::data::vector_view)` *
///         number of "inner vectors" in all jagged vectors
///       </td>
///       <td>
///         Layout metadata of the jagged vectors
///       </td>
///       <td>
///          Pointed to by `vecmem::edm::view::layout()`
///       </td>
///     </tr>
///     <tr>
///       <td>
///         Many
///       </td>
///       <td>
///         Payload of the buffer, with memory for all variables held by the
///         container
///       </td>
///       <td>
///         Pointed to by `vecmem::edm::view::payload()`
///       </td>
///     </tr>
///     <tr>
///       <th colspan="3">
///         Resizable buffer without any jagged vector variables
///       </th>
///     </tr>
///     <tr><th>Bytes</th><th>Description</th><th>Notes</th></tr>
///     <tr>
///       <td>
///         `4`
///       </td>
///       <td>
///         Size of all 1D vector variables in the container
///       </td>
///       <td>
///         Pointed to by `vecmem::edm::view::size_ptr()`
///       </td>
///     </tr>
///     <tr>
///       <td>
///         Many
///       </td>
///       <td>
///         Payload of the buffer, with memory for all variables held by the
///         container
///       </td>
///       <td>
///         Pointed to by `vecmem::edm::view::payload()`
///       </td>
///     </tr>
///     <tr>
///       <th colspan="3">
///         Resizable buffer with at least one jagged vector variable
///       </th>
///     </tr>
///     <tr><th>Bytes</th><th>Description</th><th>Notes</th></tr>
///     <tr>
///       <td>
///         `4` * number of "inner vectors" in all jagged vectors
///       </td>
///       <td>
///         Individual "inner sizes" of the jagged vectors
///       </td>
///       <td>
///         Pointed to by `vecmem::edm::view::size_ptr()`
///       </td>
///     </tr>
///     <tr>
///       <td>
///         `sizeof(vecmem::data::vector_view)` *
///         number of "inner vectors" in all jagged vectors
///       </td>
///       <td>
///         Layout metadata of the jagged vectors
///       </td>
///       <td>
///         Pointed to by `vecmem::edm::view::layout()`
///       </td>
///     </tr>
///     <tr>
///       <td>
///         Many
///       </td>
///       <td>
///         Payload of the buffer, with memory for all variables held by the
///         container
///       </td>
///       <td>
///         Pointed to by `vecmem::edm::view::payload()`
///       </td>
///     </tr>
///   </table>
///
/// The "host memory block", if used holds the same layout data for the jagged
/// vector variables as the "main memory block", just in a host-accessible
/// location. This allocation is pointed to by
/// `vecmem::edm::view::host_layout()`.
///
/// @tparam ...VARTYPES The variable types to store in the buffer
///
template <typename... VARTYPES>
class buffer<schema<VARTYPES...>> : public view<schema<VARTYPES...>> {

    // Make sure that all variable types are supported. It needs to be possible
    // to copy the contents of all variables with simple memory copies, and
    // it has to be possible to trivially destruct all objects.
    static_assert(
        std::conjunction<
            std::is_trivially_destructible<typename VARTYPES::type>...>::value,
        "Unsupported variable type");
    static_assert(std::conjunction<std::is_trivially_assignable<
                      std::add_lvalue_reference_t<typename VARTYPES::type>,
                      typename VARTYPES::type>...>::value,
                  "Unsupported variable type");

public:
    /// The schema describing the buffer's payload
    using schema_type = schema<VARTYPES...>;
    /// Base view type
    using view_type = view<schema_type>;
    /// Size type used for the container
    using size_type = typename view_type::size_type;
    /// Type holding on to the memory managed by this object
    using memory_type = unique_alloc_ptr<char[]>;

    /// Default constructor
    buffer() = default;
    /// Move constructor
    buffer(buffer&&) = default;

    /// Move assignment operator
    buffer& operator=(buffer&&) = default;

    /// Constructor for a 1D buffer
    ///
    /// @param capacity The capacity of the 1D arrays in the buffer
    /// @param mr       The memory resource to use for the allocation
    /// @param type     The type of the buffer (fixed or variable size)
    ///
    VECMEM_HOST
    buffer(
        size_type capacity, memory_resource& mr,
        vecmem::data::buffer_type type = vecmem::data::buffer_type::fixed_size);

    /// Constructor for a 2D buffer
    ///
    /// Note that the inner sizes of the jagged vector variables are technically
    /// allowed to be different. But this constructor sets them all up to have
    /// the same inner capacities.
    ///
    /// The 1D vectors of the buffer are set up to be non-resizable, with their
    /// sizes taken from @c capacities.size().
    ///
    /// @param capacities The capacities of the 1D/2D arrays in the buffer
    /// @param mr         The (main) memory resource to use for the allocation
    /// @param host_mr    The memory resource to use for the host allocation(s)
    /// @param type       The type of the buffer (fixed or variable size)
    ///
    template <typename SIZE_TYPE = std::size_t,
              std::enable_if_t<std::is_integral<SIZE_TYPE>::value &&
                                   std::is_unsigned<SIZE_TYPE>::value,
                               bool> = true>
    VECMEM_HOST buffer(
        const std::vector<SIZE_TYPE>& capacities, memory_resource& mr,
        memory_resource* host_mr = nullptr,
        vecmem::data::buffer_type type = vecmem::data::buffer_type::fixed_size);

private:
    /// Set up a fixed sized buffer
    template <typename SIZE_TYPE = std::size_t, std::size_t... INDICES>
    VECMEM_HOST void setup_fixed(const std::vector<SIZE_TYPE>& capacities,
                                 memory_resource& mr, memory_resource* host_mr,
                                 std::index_sequence<INDICES...>);
    /// Set up a resizable buffer
    template <typename SIZE_TYPE = std::size_t, std::size_t... INDICES>
    VECMEM_HOST void setup_resizable(const std::vector<SIZE_TYPE>& capacities,
                                     memory_resource& mr,
                                     memory_resource* host_mr,
                                     std::index_sequence<INDICES...>);

    /// The full allocated block of (device) memory
    memory_type m_memory;
    /// The full allocated block of host accessible memory
    memory_type m_host_memory;

};  // class buffer

}  // namespace edm

/// Helper function for getting a (possibly non-const) view for a buffer
///
/// @tparam ...VARTYPES The variable types describing the container
/// @param buffer The buffer to get a view for
/// @return A (possibly non-const) view into for the buffer
///
template <typename... VARTYPES>
VECMEM_HOST edm::view<edm::schema<VARTYPES...>> get_data(
    edm::buffer<edm::schema<VARTYPES...>>& buffer);

/// Helper function for getting a (const) view for a buffer
///
/// @tparam ...VARTYPES The variable types describing the container
/// @param buffer The buffer to get a view for
/// @return A (const) view into for the buffer
///
template <typename... VARTYPES>
VECMEM_HOST edm::view<edm::details::add_const_t<edm::schema<VARTYPES...>>>
get_data(const edm::buffer<edm::schema<VARTYPES...>>& buffer);

}  // namespace vecmem

// Include the implementation.
#include "vecmem/edm/impl/buffer.ipp"
