/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// detray core
#include "detray/definitions/detail/indexing.hpp"
#include "detray/grids/axis.hpp"
#include "detray/grids/grid2.hpp"
#include "detray/grids/populator.hpp"
#include "detray/grids/serializer2.hpp"

// Detray test include(s)
#include "detray/test/utils/types.hpp"

namespace detray {

static constexpr int n_points = 3;

using host_grid2_replace =
    grid2<replace_populator, axis2::regular, axis2::regular, serializer2,
          vecmem::vector, vecmem::jagged_vector, darray, std::tuple,
          test::point3>;

using device_grid2_replace =
    grid2<replace_populator, axis2::regular, axis2::regular, serializer2,
          vecmem::device_vector, vecmem::jagged_device_vector, darray,
          std::tuple, test::point3>;

using host_grid2_replace_ci =
    grid2<replace_populator, axis2::circular, axis2::irregular, serializer2,
          vecmem::vector, vecmem::jagged_vector, darray, std::tuple,
          test::point3>;

using device_grid2_replace_ci =
    grid2<replace_populator, axis2::circular, axis2::irregular, serializer2,
          vecmem::device_vector, vecmem::jagged_device_vector, darray,
          std::tuple, test::point3>;

using host_grid2_complete =
    grid2<complete_populator, axis2::regular, axis2::regular, serializer2,
          vecmem::vector, vecmem::jagged_vector, darray, std::tuple,
          test::point3, false, n_points>;

using device_grid2_complete =
    grid2<complete_populator, axis2::regular, axis2::regular, serializer2,
          vecmem::device_vector, vecmem::jagged_device_vector, darray,
          std::tuple, test::point3, false, n_points>;

using host_grid2_attach =
    grid2<attach_populator, axis2::circular, axis2::regular, serializer2,
          vecmem::vector, vecmem::jagged_vector, darray, std::tuple,
          test::point3, false>;

using device_grid2_attach =
    grid2<attach_populator, axis2::circular, axis2::regular, serializer2,
          vecmem::device_vector, vecmem::jagged_device_vector, darray,
          std::tuple, test::point3, false>;

using const_device_grid2_attach =
    grid2<attach_populator, axis2::circular, axis2::regular, serializer2,
          vecmem::device_vector, vecmem::jagged_device_vector, darray,
          std::tuple, const test::point3, false>;

// test function for replace populator
void grid_replace_test(grid2_view<host_grid2_replace> grid_view);

// test function for replace populator with circular and irregular axis
void grid_replace_ci_test(grid2_view<host_grid2_replace_ci> grid_view);

// test function for complete populator
void grid_complete_test(grid2_view<host_grid2_complete> grid_view);

// read test function for grid with attach populator
void grid_attach_read_test(const_grid2_view<host_grid2_attach> grid_view);

// fill test function for grid buffer with attach populator
void grid_attach_fill_test(grid2_view<host_grid2_attach> grid_view);

// assign test function for grid buffer with attach populator
void grid_attach_assign_test(grid2_view<host_grid2_attach> grid_view);

// read test function for grid array
template <template <typename, size_t> class array_type>
void grid_array_test(array_type<grid2_view<host_grid2_attach>, 2> grid_array,
                     vecmem::data::vector_view<test::point3>& outputs_data);

}  // namespace detray
