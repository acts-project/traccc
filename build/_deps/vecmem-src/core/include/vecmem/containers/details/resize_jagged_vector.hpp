/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/jagged_vector.hpp"
#include "vecmem/containers/vector.hpp"

namespace vecmem::details {

/// Resize a generic jagged vector
///
/// It does exactly what @c std::vector::resize does.
///
/// @param vec The vector to resize
/// @param size The size to resize the jagged vector to
///
template <typename T, typename ALLOC1, typename ALLOC2>
void resize_jagged_vector(std::vector<std::vector<T, ALLOC1>, ALLOC2>& vec,
                          std::size_t size) {
    vec.resize(size);
}

/// Resize a vecmem jagged vector
///
/// It makes sure that all of the "internal" vectors would use the same memory
/// resource as the "external" one does.
///
/// @param vec The vector to resize
/// @param size The size to resize the jagged vector to
///
template <typename T>
void resize_jagged_vector(jagged_vector<T>& vec, std::size_t size) {
    vec.resize(size, vecmem::vector<T>(vec.get_allocator().resource()));
}

}  // namespace vecmem::details
