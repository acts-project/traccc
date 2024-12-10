/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// detray core
#include "detray/definitions/detail/indexing.hpp"
#include "detray/utils/ranges.hpp"

// vecmem core
#include "vecmem/containers/device_vector.hpp"

namespace detray {

/// Test type
struct uint_holder {
    dindex ui = 0u;
};

/// Test @c detray::views::single
void test_single(const dindex value, dindex& check);

/// Test @c detray::views::pointer
void test_pointer(const dindex value, dindex& check);

/// Test @c detray::views::iota
void test_iota(const darray<dindex, 2> range,
               vecmem::data::vector_view<dindex> check_data);

/// Test @c detray::views::cartesian_product
void test_cartesian_product(
    const darray<dindex, 2> range1, const darray<dindex, 2> range2,
    const darray<dindex, 2> range3,
    vecmem::data::vector_view<std::tuple<dindex, dindex, dindex>> check_data);

/// Test @c detray::views::enumerate
void test_enumerate(vecmem::data::vector_view<uint_holder> seq_data,
                    vecmem::data::vector_view<dindex> check_idx_data,
                    vecmem::data::vector_view<dindex> check_value_data);

/// Test @c detray::views::pick
void test_pick(vecmem::data::vector_view<uint_holder> seq_data,
               vecmem::data::vector_view<dindex> idx_data,
               vecmem::data::vector_view<dindex> check_idx_data,
               vecmem::data::vector_view<dindex> check_value_data);

/// Test @c detray::views::join
void test_join(vecmem::data::vector_view<uint_holder> seq_data_1,
               vecmem::data::vector_view<uint_holder> seq_data_2,
               vecmem::data::vector_view<dindex> check_value_data);

/// Test @c detray::views::static_join
void test_static_join(vecmem::data::vector_view<uint_holder> seq_data_1,
                      vecmem::data::vector_view<uint_holder> seq_data_2,
                      vecmem::data::vector_view<dindex> check_value_data);

/// Test @c detray::views::subrange
void test_subrange(vecmem::data::vector_view<int> seq_data,
                   vecmem::data::vector_view<int> check_value_data,
                   const std::size_t begin, const std::size_t end);

}  // namespace detray
