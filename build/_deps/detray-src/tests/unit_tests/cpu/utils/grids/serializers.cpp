/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// detray core
#include "detray/utils/grid/serializers.hpp"

#include "detray/definitions/detail/indexing.hpp"
#include "detray/geometry/coordinates/cylindrical3D.hpp"
#include "detray/geometry/coordinates/polar2D.hpp"
#include "detray/geometry/mask.hpp"
#include "detray/utils/grid/detail/axis.hpp"

// Detray test include(s)
#include "detray/test/utils/types.hpp"

// vecmem include(s)
#include <vecmem/containers/vector.hpp>

// GTest include(s)
#include <gtest/gtest.h>

// System inculde(s)
#include <climits>

using namespace detray;
using namespace detray::axis;

namespace {

// Axes types to be serialized

// polar coordinate system with regular binning on both axes
using polar_axes = multi_axis<
    true, polar2D<test::algebra>,
    single_axis<closed<label::e_r>, regular<host_container_types, scalar>>,
    single_axis<circular<label::e_phi>, regular<host_container_types, scalar>>>;
// 3-dim cylindrical coordinate system with regular binning
using cylinder_axes = multi_axis<
    true, cylindrical3D<test::algebra>,
    single_axis<closed<label::e_r>, regular<host_container_types, scalar>>,
    single_axis<circular<label::e_phi>, regular<host_container_types, scalar>>,
    single_axis<closed<label::e_z>, regular<host_container_types, scalar>>>;

}  // anonymous namespace

GTEST_TEST(detray_grid, serializer2D) {

    // Offsets into edges container and #bins for all axes
    vecmem::vector<dindex_range> edge_ranges = {{0u, 6u}, {2u, 12u}};
    // Not needed for serializer test
    vecmem::vector<scalar> bin_edges{};

    polar_axes axes(std::move(edge_ranges), std::move(bin_edges));

    simple_serializer<2> serializer{};

    // Serializing
    multi_bin<2> mbin{0u, 0u};
    EXPECT_EQ(serializer(axes, mbin), 0u);
    mbin = {5u, 0u};
    EXPECT_EQ(serializer(axes, mbin), 5u);
    mbin = {0u, 1u};
    EXPECT_EQ(serializer(axes, mbin), 6u);
    mbin = {5u, 2u};
    EXPECT_EQ(serializer(axes, mbin), 17u);

    // Deserialize
    multi_bin<2> expected_mbin{0u, 0u};
    EXPECT_EQ(serializer(axes, 0u), expected_mbin);
    expected_mbin = {5u, 0u};
    EXPECT_EQ(serializer(axes, 5u), expected_mbin);
    expected_mbin = {0u, 1u};
    EXPECT_EQ(serializer(axes, 6u), expected_mbin);
    expected_mbin = {5u, 2u};
    EXPECT_EQ(serializer(axes, 17u), expected_mbin);
}

GTEST_TEST(detray_grid, serializer3D) {

    // Offsets into edges container and #bins for all axes
    vecmem::vector<dindex_range> edge_ranges = {{0u, 4u}, {2u, 2u}, {4u, 2u}};
    // Not needed for serializer test
    vecmem::vector<scalar> bin_edges{};

    cylinder_axes axes(std::move(edge_ranges), std::move(bin_edges));

    simple_serializer<3> serializer{};

    // Serializing
    multi_bin<3> mbin{0u, 0u, 0u};
    EXPECT_EQ(serializer(axes, mbin), 0u);
    mbin = {2u, 1u, 0u};
    EXPECT_EQ(serializer(axes, mbin), 6u);
    mbin = {3u, 0u, 1u};
    EXPECT_EQ(serializer(axes, mbin), 11u);
    mbin = {1u, 1u, 1u};
    EXPECT_EQ(serializer(axes, mbin), 13u);

    // Deserialize
    multi_bin<3> expected_mbin{0u, 0u, 0u};
    EXPECT_EQ(serializer(axes, 0u), expected_mbin);
    expected_mbin = {2u, 1u, 0u};
    EXPECT_EQ(serializer(axes, 6u), expected_mbin);
    expected_mbin = {3u, 0u, 1u};
    EXPECT_EQ(serializer(axes, 11u), expected_mbin);
    expected_mbin = {1u, 1u, 1u};
    EXPECT_EQ(serializer(axes, 13u), expected_mbin);
}
