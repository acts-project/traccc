/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem {
namespace data {

/// @brief "Overall type" for a buffer object
enum class buffer_type {

    fixed_size = 0,  ///< The buffer has a fixed number of elements
    resizable = 1    ///< The buffer is resizable/expandable

};  // enum class buffer_type

}  // namespace data
}  // namespace vecmem
