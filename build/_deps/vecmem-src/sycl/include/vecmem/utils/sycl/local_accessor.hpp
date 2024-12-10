/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Cowardly don't include <sycl/sycl.hpp> here, leave it up to the user to
// do that. This way the vecmem::sycl library doesn't have to set up an
// explicit public dependency on the SYCL headers.

namespace vecmem::sycl {

#if defined(VECMEM_HAVE_SYCL_LOCAL_ACCESSOR) && \
    (defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION))

/// @brief Set up @c vecmem::sycl::local_accessor as an alias for
///       @c ::sycl::local_accessor.
/// @tparam T The type of the local memory array.
/// @tparam DIM Dimensions for the local memory array.
///
template <typename T, int DIM = 1>
using local_accessor = ::sycl::local_accessor<T, DIM>;

#elif (defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION))

/// @brief Set up @c vecmem::sycl::local_accessor as an alias for
///       @c ::sycl::accessor.
/// @tparam T The type of the local memory array.
/// @tparam DIM Dimensions for the local memory array.
///
template <typename T, int DIM = 1>
using local_accessor =
    ::sycl::accessor<T, DIM, ::sycl::access::mode::read_write,
                     ::sycl::access::target::local>;

#endif  // VECMEM_HAVE_SYCL_LOCAL_ACCESSOR

}  // namespace vecmem::sycl
