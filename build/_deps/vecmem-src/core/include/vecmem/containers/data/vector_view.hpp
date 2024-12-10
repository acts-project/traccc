/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/utils/type_traits.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <cstddef>
#include <type_traits>

namespace vecmem {

/// @brief Namespace holding "data types"
///
/// These are types that either own, or only provide a view of data owned by
/// some other component. They are used for "non-interactive" data management
/// in the code.
///
namespace data {

/// Class holding data about a 1 dimensional vector/array
///
/// This type is meant to "formalise" the communication of data between
/// @c vecmem::vector, @c vecmem::array ("host types") and
/// @c vecmem::(const_)device_vector, @c vecmem::(const_)device_array
/// ("device types").
///
/// This type does not own the data that it points to. It merely provides a
/// "view" of that data.
///
template <typename TYPE>
class vector_view {

    /// We cannot use boolean types.
    static_assert(
        !std::is_same<typename std::remove_cv<TYPE>::type, bool>::value,
        "bool is not supported in VecMem containers");

public:
    /// Size type used in the class
    typedef unsigned int size_type;
    /// Pointer type to the size of the array
    typedef
        typename std::conditional<std::is_const<TYPE>::value, const size_type*,
                                  size_type*>::type size_pointer;
    /// Constant pointer type to the size of the array
    typedef const typename std::remove_const<size_pointer>::type
        const_size_pointer;
    /// Pointer type to the array
    typedef TYPE* pointer;
    /// Constant pointer to the array
    typedef const typename std::remove_const<pointer>::type const_pointer;

    /// Default constructor
    vector_view() = default;
    /// Constant size data constructor
    VECMEM_HOST_AND_DEVICE
    vector_view(size_type size, pointer ptr);
    /// Resizable data constructor
    VECMEM_HOST_AND_DEVICE
    vector_view(size_type capacity, size_pointer size, pointer ptr);

    /// Constructor from a "slightly different" @c vecmem::details::vector_view
    /// object
    ///
    /// Only enabled if the wrapped type is different, but only by const-ness.
    /// This complication is necessary to avoid problems from SYCL. Which is
    /// very particular about having default copy constructors for the types
    /// that it sends to kernels.
    ///
    template <typename OTHERTYPE,
              std::enable_if_t<details::is_same_nc<TYPE, OTHERTYPE>::value,
                               bool> = true>
    VECMEM_HOST_AND_DEVICE vector_view(const vector_view<OTHERTYPE>& parent);

    /// Copy from a "slightly different" @c vecmem::details::vector_view object
    ///
    /// See the copy constructor for more details.
    ///
    template <typename OTHERTYPE,
              std::enable_if_t<details::is_same_nc<TYPE, OTHERTYPE>::value,
                               bool> = true>
    VECMEM_HOST_AND_DEVICE vector_view& operator=(
        const vector_view<OTHERTYPE>& rhs);

    /// Equality check. Two objects are only equal if they point at the same
    /// memory.
    template <
        typename OTHERTYPE,
        std::enable_if_t<std::is_same<std::remove_cv_t<TYPE>,
                                      std::remove_cv_t<OTHERTYPE> >::value,
                         bool> = true>
    VECMEM_HOST_AND_DEVICE bool operator==(
        const vector_view<OTHERTYPE>& rhs) const;

    /// Inequality check. Simply based on @c operator==.
    template <
        typename OTHERTYPE,
        std::enable_if_t<std::is_same<std::remove_cv_t<TYPE>,
                                      std::remove_cv_t<OTHERTYPE> >::value,
                         bool> = true>
    VECMEM_HOST_AND_DEVICE bool operator!=(
        const vector_view<OTHERTYPE>& rhs) const;

    /// Get the size of the vector
    VECMEM_HOST_AND_DEVICE
    size_type size() const;
    /// Get the maximum capacity of the vector
    VECMEM_HOST_AND_DEVICE
    size_type capacity() const;

    /// Get a pointer to the size of the vector (non-const)
    VECMEM_HOST_AND_DEVICE
    size_pointer size_ptr();
    /// Get a pointer to the size of the vector (const)
    VECMEM_HOST_AND_DEVICE
    const_size_pointer size_ptr() const;

    /// Get a pointer to the vector elements (non-const)
    VECMEM_HOST_AND_DEVICE
    pointer ptr();
    /// Get a pointer to the vector elements (const)
    VECMEM_HOST_AND_DEVICE
    const_pointer ptr() const;

protected:
    /// Maximum capacity of the array
    size_type m_capacity;
    /// Pointer to the size of the array in memory
    size_pointer m_size;
    /// Pointer to the start of the memory block/array
    pointer m_ptr;

};  // struct vector_view

}  // namespace data
}  // namespace vecmem

// Include the implementation.
#include "vecmem/containers/impl/vector_view.ipp"
