/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/concepts.hpp"

#if __cpp_concepts >= 201907L
#include <concepts>
#endif

namespace traccc::device::concepts {
#if __cpp_concepts >= 201907L
/**
 * @brief Concept to ensure that a type behaves like a thread identification
 * type which allows us to access thread and block IDs. This concept assumes
 * one-dimensional grids.
 *
 * @tparam T The thread identifier-like type.
 */
template <typename T>
concept thread_id1 = requires(T& i) {
    /*
     * This function should return the local thread identifier in a *flat* way,
     * e.g. compressing two or three dimensional blocks into one dimension.
     */
    { i.getLocalThreadId() }
    ->std::integral;

    /*
     * This function should return the local thread identifier in the X-axis.
     */
    { i.getLocalThreadIdX() }
    ->std::integral;

    /*
     * This function should return the global thread identifier in a *flat*
     * way, e.g. compressing two or three dimensional blocks into one
     * dimension.
     */
    { i.getGlobalThreadId() }
    ->std::integral;

    /*
     * This function should return the global thread identifier in the X-axis.
     */
    { i.getGlobalThreadIdX() }
    ->std::integral;

    /*
     * This function should return the block identifier in the X-axis.
     */
    { i.getBlockIdX() }
    ->std::integral;

    /*
     * This function should return the block size in the X-axis.
     */
    { i.getBlockIdX() }
    ->std::integral;

    /*
     * This function should return the grid identifier in the X-axis.
     */
    { i.getBlockIdX() }
    ->std::integral;
};
#endif
}  // namespace traccc::device::concepts
