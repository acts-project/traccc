/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <memory>
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// System include
#include <concepts>
#include <memory>
#include <unordered_set>

namespace traccc::host {
/**
 * @brief Sanity check that a given vector is contiguous on a given projection.
 *
 * For a vector $v$ to be contiguous on a projection $\pi$, it must be the case
 * that for all indices $i$ and $j$, if $v_i = v_j$, then all indices $k$
 * between $i$ and $j$, $v_i = v_j = v_k$.
 *
 * @note This function runs in O(n^2) time.
 *
 * @tparam P The type of projection $\pi$, a callable which returns some
 * comparable type.
 * @tparam T The type of the vector.
 * @param projection A projection object of type `P`.
 * @param mr A memory resource used for allocating intermediate memory.
 * @param vector The vector which to check for contiguity.
 * @return true If the vector is contiguous on `P`.
 * @return false Otherwise.
 */
template <std::semiregular P, std::equality_comparable T>
requires std::regular_invocable<P, T> bool is_contiguous_on(
    P&& projection, vecmem::data::vector_view<T> vector) {
    // Grab the number of elements in our vector.
    uint32_t n = vector.size();

    // Get the output type of the projection.
    using projection_t = std::invoke_result_t<P, T>;

    // Allocate memory for intermediate values and outputs, then set them up.
    std::unique_ptr<projection_t[]> iout = std::make_unique<projection_t[]>(n);
    uint32_t iout_size = 0;

    // Create a device vector
    vecmem::device_vector<T> in(vector);

    // Compress adjacent elements
    for (std::size_t i = 0; i < n; ++i) {
        if (i == 0) {
            iout[iout_size++] = projection(in.at(i));
        } else {
            projection_t v = projection(in.at(i));

            if (v != iout[iout_size - 1]) {
                iout[iout_size++] = v;
            }
        }
    }

    // Check whether all elements are unique
    std::unordered_set<projection_t> seen;

    for (std::size_t i = 0; i < iout_size; ++i) {
        projection_t& v = iout[i];

        if (seen.count(v) == 1) {
            return false;
        } else {
            seen.insert(v);
        }
    }

    return true;
}
}  // namespace traccc::host
