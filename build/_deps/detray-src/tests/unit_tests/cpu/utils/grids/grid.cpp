/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Detray include(s)
#include "detray/utils/grid/grid.hpp"

#include "detray/builders/grid_builder.hpp"
#include "detray/definitions/detail/indexing.hpp"
#include "detray/geometry/shapes/cuboid3D.hpp"
#include "detray/utils/grid/detail/concepts.hpp"

// Detray test include(s)
#include "detray/test/utils/types.hpp"

// Vecmem include(s)
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s)
#include <gtest/gtest.h>

// System include(s)
#include <algorithm>
#include <limits>
#include <random>

using namespace detray;
using namespace detray::axis;

namespace {

// Algebra definitions
using point3 = test::point3;

constexpr scalar inf{std::numeric_limits<scalar>::max()};
constexpr scalar tol{1e-7f};

// Either a data owning or non-owning 3D cartesian multi-axis
template <bool ownership = true, typename containers = host_container_types>
using cartesian_3D = coordinate_axes<axes<cuboid3D>, ownership, containers>;

// non-owning multi-axis: Takes external containers
bool constexpr is_owning = true;
bool constexpr is_n_owning = false;

// Bin edges for all axes
dvector<scalar> bin_edges = {-10.f, 10.f, -20.f, 20.f, 0.f, 100.f};
// Offsets into edges container and #bins for all axes
dvector<dindex_range> edge_ranges = {{0u, 20u}, {2u, 40u}, {4u, 50u}};

// non-owning multi-axis for the non-owning grid
cartesian_3D<is_n_owning, host_container_types> ax_n_own(edge_ranges,
                                                         bin_edges);

// Create some bin data for non-owning grid
template <typename populator_t, typename bin_t>
struct bin_content_sequence {
    using entry_t = typename bin_t::entry_type;
    entry_t entry{0};

    auto operator()() {
        entry += entry_t{1};
        bin_t bin{};
        populator_t{}(bin, entry);
        return bin;
    }
};

/// Test bin content element by element
template <concepts::grid grid_t, typename content_t>
void test_content(const grid_t& g, const point3& p, const content_t& expected) {
    dindex i = 0u;
    for (const auto& entry : g.search(p)) {
        ASSERT_NEAR(entry, expected[i++], tol) << " at index " << i - 1u;
    }
}

}  // anonymous namespace

/// Unittest: Test single grid construction
GTEST_TEST(detray_grid, single_grid) {

    // Owning and non-owning, cartesian, 3-dimensional grids
    using grid_owning_t = grid<axes<cuboid3D>, bins::single<scalar>>;

    using grid_n_owning_t =
        grid<axes<cuboid3D>, bins::single<scalar>, simple_serializer,
             host_container_types, false>;

    using grid_device_t = grid<axes<cuboid3D>, bins::single<scalar>,
                               simple_serializer, device_container_types>;

    static_assert(concepts::grid<grid_owning_t>);
    static_assert(concepts::grid<grid_n_owning_t>);
    static_assert(concepts::grid<grid_device_t>);

    // Fill the bin data for every test
    // bin test entries
    grid_owning_t::bin_container_type bin_data{};
    bin_data.resize(40'000u);
    std::ranges::generate_n(
        bin_data.begin(), 40'000u,
        bin_content_sequence<replace<>, bins::single<scalar>>());

    // Copy data that will be moved into the data owning types
    dvector<scalar> bin_edges_cp(bin_edges);
    dvector<dindex_range> edge_ranges_cp(edge_ranges);
    grid_owning_t::bin_container_type bin_data_cp(bin_data);

    // Data-owning axes and grid
    cartesian_3D<is_owning, host_container_types> axes_own(
        std::move(edge_ranges_cp), std::move(bin_edges_cp));
    grid_owning_t grid_own(std::move(bin_data_cp), std::move(axes_own));

    // Copy a second time for the comparison
    dvector<scalar> bin_edges_cp2(bin_edges);
    dvector<dindex_range> edge_ranges_cp2(edge_ranges);
    grid_owning_t::bin_container_type bin_data_cp2(bin_data);

    // Make a second grid
    cartesian_3D<is_owning, host_container_types> axes_own2(
        std::move(edge_ranges_cp2), std::move(bin_edges_cp2));
    grid_owning_t grid_own2(std::move(bin_data_cp2), std::move(axes_own2));

    // CHECK equality
    EXPECT_TRUE(grid_own == grid_own2);

    // Check a few basics
    EXPECT_EQ(grid_own.dim, 3u);
    EXPECT_EQ(grid_own.nbins(), 40'000u);
    auto y_axis = grid_own.get_axis<label::e_y>();
    EXPECT_EQ(y_axis.nbins(), 40u);
    auto z_axis =
        grid_own.get_axis<single_axis<closed<label::e_z>, regular<>>>();
    EXPECT_EQ(z_axis.nbins(), 50u);

    // Create non-owning grid
    grid_n_owning_t grid_n_own(&bin_data, ax_n_own);

    // Test for consistency with owning grid
    EXPECT_EQ(grid_n_own.dim, grid_own.dim);
    y_axis = grid_n_own.get_axis<label::e_y>();
    EXPECT_EQ(y_axis.nbins(), grid_own.get_axis<label::e_y>().nbins());
    z_axis = grid_n_own.get_axis<label::e_z>();
    EXPECT_EQ(z_axis.nbins(), grid_own.get_axis<label::e_z>().nbins());

    // Construct a grid from a view
    grid_owning_t::view_type grid_view = get_data(grid_own);
    grid_device_t device_grid(grid_view);

    // Test for consistency with non-owning grid
    EXPECT_EQ(device_grid.dim, grid_n_own.dim);
    auto y_axis_dev = device_grid.get_axis<label::e_y>();
    EXPECT_EQ(y_axis_dev.nbins(), grid_n_own.get_axis<label::e_y>().nbins());
    auto z_axis_dev = device_grid.get_axis<label::e_z>();
    EXPECT_EQ(z_axis_dev.nbins(), grid_n_own.get_axis<label::e_z>().nbins());

    // Test the global bin iteration: owning grid
    auto seq = detray::views::iota(1, 40'001);
    auto flat_bin_view = grid_own.all();

    static_assert(detray::ranges::random_access_range<decltype(flat_bin_view)>);

    EXPECT_EQ(seq.size(), 40'000u);
    EXPECT_EQ(flat_bin_view.size(), 40'000u);
    EXPECT_EQ(flat_bin_view[42], 43u);
    EXPECT_TRUE(
        std::equal(flat_bin_view.begin(), flat_bin_view.end(), seq.begin()));

    // Test the global bin iteration: non-owning grid
    auto flat_bin_view2 = grid_n_own.all();

    static_assert(
        detray::ranges::random_access_range<decltype(flat_bin_view2)>);

    EXPECT_EQ(seq.size(), 40'000u);
    EXPECT_EQ(flat_bin_view2.size(), 40'000u);
    EXPECT_EQ(flat_bin_view2[42], 43u);
    EXPECT_TRUE(
        std::equal(flat_bin_view2.begin(), flat_bin_view2.end(), seq.begin()));

    // Test const grid view
    /*auto const_grid_view = get_data(const_cast<const
    grid_owning_t&>(grid_own));

    static_assert(
        std::is_same_v<decltype(const_grid_view),
                       typename grid<cartesian_3D<>, bins::single<const
    scalar>>::view_type>, "Const grid view was not correctly constructed!");

    grid<cartesian_3D<is_owning, device_container_types>, bins::single<const
    scalar>> const_device_grid(const_grid_view);

    static_assert(
        std::is_same_v<typename decltype(const_device_grid)::bin_type,
                       typename replace::template bin_type<const scalar>>,
        "Const grid was not correctly constructed from view!");*/
}

/// Unittest: Test dynamic grid construction
GTEST_TEST(detray_grid, dynamic_array) {

    // Owning and non-owning, cartesian, 3-dimensional grids
    using grid_owning_t = grid<axes<cuboid3D>, bins::dynamic_array<scalar>>;

    using grid_n_owning_t =
        grid<axes<cuboid3D>, bins::dynamic_array<scalar>, simple_serializer,
             host_container_types, false>;

    using grid_device_t = grid<axes<cuboid3D>, bins::dynamic_array<scalar>,
                               simple_serializer, device_container_types>;

    static_assert(concepts::grid<grid_owning_t>);
    static_assert(concepts::grid<grid_n_owning_t>);
    static_assert(concepts::grid<grid_device_t>);

    // Fill the bin data for every test
    // bin test entries
    grid_owning_t::bin_container_type bin_data{};
    // 40 000 entries
    bin_data.entries.resize(80'000u);
    // 20 000 bins
    bin_data.bins.resize(40'000u);

    int i{0};
    dindex offset{0u};
    scalar entry{0.f};
    complete<> completer{};

    // Test data to compare bin content against
    std::vector<scalar> seq;
    seq.reserve(80'000);

    for (auto& data : bin_data.bins) {
        data.offset = offset;
        // Every second bin holds one element, otherwise three
        data.capacity = (i % 2) ? 1u : 3u;

        detray::bins::dynamic_array bin{bin_data.entries.data(), data};

        ASSERT_TRUE(bin.capacity() == (i % 2 ? 1u : 3u));
        ASSERT_TRUE(bin.size() == 0);

        offset += bin.capacity();

        // Populate the bin
        completer(bin, entry);

        for (auto e : bin) {
            ASSERT_TRUE(e == entry);
            seq.push_back(e);
        }
        entry += 1.f;
        ++i;
    }

    // Copy data that will be moved into the data owning types
    dvector<scalar> bin_edges_cp(bin_edges);
    dvector<dindex_range> edge_ranges_cp(edge_ranges);
    grid_owning_t::bin_container_type bin_data_cp(bin_data);

    // Data-owning axes and grid
    cartesian_3D<is_owning, host_container_types> axes_own(
        std::move(edge_ranges_cp), std::move(bin_edges_cp));
    grid_owning_t grid_own(std::move(bin_data_cp), std::move(axes_own));

    // Check a few basics
    EXPECT_EQ(grid_own.dim, 3u);
    EXPECT_EQ(grid_own.nbins(), 40'000u);
    auto y_axis = grid_own.get_axis<label::e_y>();
    EXPECT_EQ(y_axis.nbins(), 40u);
    auto z_axis =
        grid_own.get_axis<single_axis<closed<label::e_z>, regular<>>>();
    EXPECT_EQ(z_axis.nbins(), 50u);

    // Check equality operator:
    // - Copy a second time for the comparison
    dvector<scalar> bin_edges_cp2(bin_edges);
    dvector<dindex_range> edge_ranges_cp2(edge_ranges);
    grid_owning_t::bin_container_type bin_data_cp2(bin_data);

    // Make a second grid
    cartesian_3D<is_owning, host_container_types> axes_own2(
        std::move(edge_ranges_cp2), std::move(bin_edges_cp2));

    grid_owning_t grid_own2(std::move(bin_data_cp2), std::move(axes_own2));

    // CHECK equality
    EXPECT_TRUE(grid_own == grid_own2);

    // Create non-owning grid
    grid_n_owning_t grid_n_own(&bin_data, ax_n_own);

    // Test for consistency with owning grid
    EXPECT_EQ(grid_n_own.dim, grid_own.dim);
    y_axis = grid_n_own.get_axis<label::e_y>();
    EXPECT_EQ(y_axis.nbins(), grid_own.get_axis<label::e_y>().nbins());
    z_axis = grid_n_own.get_axis<label::e_z>();
    EXPECT_EQ(z_axis.nbins(), grid_own.get_axis<label::e_z>().nbins());

    // Construct a grid from a view
    grid_owning_t::view_type grid_view = get_data(grid_own);
    grid_device_t device_grid(grid_view);

    // Test for consistency with non-owning grid
    EXPECT_EQ(device_grid.dim, grid_n_own.dim);
    auto y_axis_dev = device_grid.get_axis<label::e_y>();
    EXPECT_EQ(y_axis_dev.nbins(), grid_n_own.get_axis<label::e_y>().nbins());
    auto z_axis_dev = device_grid.get_axis<label::e_z>();
    EXPECT_EQ(z_axis_dev.nbins(), grid_n_own.get_axis<label::e_z>().nbins());

    // Test the global bin iteration
    auto flat_bin_view = grid_own.all();

    static_assert(detray::ranges::bidirectional_range<decltype(flat_bin_view)>);
    // TODO: Const-correctness issue
    // static_assert(detray::ranges::random_access_range<typename decltype(
    //                  flat_bin_view)>);

    EXPECT_EQ(seq.size(), 80'000u);
    EXPECT_EQ(flat_bin_view.size(), 80'000u);
    EXPECT_TRUE(
        std::equal(flat_bin_view.begin(), flat_bin_view.end(), seq.begin()));

    // Test the global bin iteration: non-owning grid
    auto flat_bin_view2 = grid_n_own.all();

    static_assert(
        detray::ranges::bidirectional_range<decltype(flat_bin_view2)>);

    EXPECT_EQ(seq.size(), 80'000u);
    EXPECT_EQ(flat_bin_view2.size(), 80'000u);
    EXPECT_TRUE(
        std::equal(flat_bin_view2.begin(), flat_bin_view2.end(), seq.begin()));
}

/// Test bin entry retrieval
GTEST_TEST(detray_grid, bin_view) {

    // Non-owning, 3D cartesian, replacing grid
    using grid_t = grid<axes<cuboid3D>, bins::single<scalar>>;

    static_assert(concepts::grid<grid_t>);

    // Fill the bin data for every test
    // bin test entries
    grid_t::bin_container_type bin_data{};
    bin_data.resize(40'000u);
    std::ranges::generate_n(
        bin_data.begin(), 40'000u,
        bin_content_sequence<replace<>, bins::single<scalar>>());

    // Copy data that will be moved into the data owning types
    dvector<scalar> bin_edges_cp(bin_edges);
    dvector<dindex_range> edge_ranges_cp(edge_ranges);
    grid_t::bin_container_type bin_data_cp(bin_data);

    // Data-owning axes and grid
    cartesian_3D<is_owning, host_container_types> axes_own(
        std::move(edge_ranges_cp), std::move(bin_edges_cp));
    grid_t grid_3D(std::move(bin_data_cp), std::move(axes_own));

    // Test the bin view
    point3 p = {-10.f, -20.f, 0.f};

    std::array<dindex, 2> search_window_size{0, 0};
    axis::multi_bin_range<3> search_window{
        axis::bin_range{0, 1}, axis::bin_range{0, 1}, axis::bin_range{0, 1}};

    const auto bview1 = axis::detail::bin_view(grid_3D, search_window);
    const auto joined_view1 = detray::views::join(bview1);
    const auto grid_search1 = grid_3D.search(p, search_window_size);

    static_assert(detray::ranges::bidirectional_range<decltype(bview1)>);
    static_assert(detray::ranges::bidirectional_range<decltype(joined_view1)>);
    static_assert(detray::ranges::bidirectional_range<decltype(grid_search1)>);

    ASSERT_EQ(bview1.size(), 1u);
    ASSERT_EQ(joined_view1.size(), 1u);
    ASSERT_EQ(grid_search1.size(), 1u);

    for (auto bin : bview1) {
        for (auto entry : bin) {
            EXPECT_EQ(entry, grid_3D.bin(0).value()) << "bin entry: " << entry;
        }
    }

    for (scalar entry : joined_view1) {
        EXPECT_EQ(entry, grid_3D.bin(0).value()) << "bin entry: " << entry;
    }

    for (scalar entry : grid_search1) {
        EXPECT_EQ(entry, grid_3D.bin(0).value()) << "bin entry: " << entry;
    }

    //
    // In the corner of the grid, include nearest neighbor
    //
    search_window_size[0] = 1;
    search_window_size[1] = 1;
    search_window[0] = axis::bin_range{0, 2};
    search_window[1] = axis::bin_range{0, 2};
    search_window[2] = axis::bin_range{0, 2};

    std::vector<scalar> expected{1, 801, 21, 821, 2, 802, 22, 822};

    const auto bview2 = axis::detail::bin_view(grid_3D, search_window);
    const auto joined_view2 = detray::views::join(bview2);
    const auto grid_search2 = grid_3D.search(p, search_window_size);

    static_assert(detray::ranges::bidirectional_range<decltype(bview2)>);
    static_assert(detray::ranges::bidirectional_range<decltype(joined_view2)>);
    static_assert(detray::ranges::bidirectional_range<decltype(grid_search2)>);

    ASSERT_EQ(bview2.size(), 8u);
    ASSERT_EQ(joined_view2.size(), 8u);
    ASSERT_EQ(grid_search2.size(), 8u);

    for (auto [i, bin] : detray::views::enumerate(bview2)) {
        for (scalar entry : bin) {
            EXPECT_EQ(entry, expected[i]) << "bin entry: " << entry;
        }
    }

    for (auto [i, entry] : detray::views::enumerate(joined_view2)) {
        EXPECT_EQ(entry, expected[i]) << "bin entry: " << entry;
    }

    for (auto [i, entry] : detray::views::enumerate(grid_search2)) {
        EXPECT_EQ(entry, expected[i]) << "bin entry: " << entry;
    }

    //
    // Bin 1 and nearest neighbors
    //
    p = {-9.f, -19.f, 2.f};
    search_window[0] = axis::bin_range{0, 3};
    search_window[1] = axis::bin_range{0, 3};
    search_window[2] = axis::bin_range{0, 3};

    expected = {1, 801, 1601, 21, 821, 1621, 41, 841, 1641,
                2, 802, 1602, 22, 822, 1622, 42, 842, 1642,
                3, 803, 1603, 23, 823, 1623, 43, 843, 1643};

    const auto bview3 = axis::detail::bin_view(grid_3D, search_window);
    const auto joined_view3 = detray::views::join(bview3);
    const auto grid_search3 = grid_3D.search(p, search_window_size);

    static_assert(detray::ranges::bidirectional_range<decltype(bview3)>);
    static_assert(detray::ranges::bidirectional_range<decltype(joined_view3)>);
    static_assert(detray::ranges::bidirectional_range<decltype(grid_search3)>);

    ASSERT_EQ(bview3.size(), 27u);
    ASSERT_EQ(joined_view3.size(), 27u);
    ASSERT_EQ(grid_search3.size(), 27u);

    for (auto [i, bin] : detray::views::enumerate(bview3)) {
        for (scalar entry : bin) {
            EXPECT_EQ(entry, expected[i]) << "bin entry: " << entry;
        }
    }

    for (auto [i, entry] : detray::views::enumerate(joined_view3)) {
        EXPECT_EQ(entry, expected[i]) << "bin entry: " << entry;
    }

    for (auto [i, entry] : detray::views::enumerate(grid_search3)) {
        EXPECT_EQ(entry, expected[i]) << "bin entry: " << entry;
    }
}

/// Integration test: Test replace population
GTEST_TEST(detray_grid, replace_population) {

    // Non-owning, 3D cartesian  grid
    using grid_t = grid<decltype(ax_n_own), bins::single<scalar>,
                        simple_serializer, host_container_types, false>;

    static_assert(concepts::grid<grid_t>);

    // init
    using bin_t = grid_t::bin_type;
    grid_t::bin_container_type bin_data{};
    bin_data.resize(40'000u, bin_t{});

    // Create non-owning grid
    grid_t g3r(&bin_data, ax_n_own);

    // Test the initialization
    point3 p = {-10.f, -20.f, 0.f};
    for (int ib0 = 0; ib0 < 20; ++ib0) {
        for (int ib1 = 0; ib1 < 40; ++ib1) {
            for (int ib2 = 0; ib2 < 100; ib2 += 2) {
                p = {static_cast<scalar>(-10 + ib0),
                     static_cast<scalar>(-20 + ib1),
                     static_cast<scalar>(0 + ib2)};
                EXPECT_NEAR(g3r.search(p)[0],
                            std::numeric_limits<scalar>::max(), tol);
            }
        }
    }

    p = {-4.5f, -4.5f, 4.5f};
    // Fill and read
    g3r.template populate<replace<>>(p, 3.f);
    EXPECT_NEAR(g3r.search(p)[0], static_cast<scalar>(3u), tol);

    // Fill and read two times, fill first 0-99, then 100-199
    for (unsigned int il = 0u; il < 2u; ++il) {
        scalar counter{static_cast<scalar>(il * 100u)};
        for (int ib0 = 0; ib0 < 20; ++ib0) {
            for (int ib1 = 0; ib1 < 40; ++ib1) {
                for (int ib2 = 0; ib2 < 100; ib2 += 2) {
                    p = {static_cast<scalar>(-10 + ib0),
                         static_cast<scalar>(-20 + ib1),
                         static_cast<scalar>(0 + ib2)};
                    g3r.template populate<replace<>>(p, counter);
                    EXPECT_NEAR(g3r.search(p)[0], counter, tol);
                    counter += 1.f;
                }
            }
        }
    }
}

/// Test bin entry retrieval
GTEST_TEST(detray_grid, complete_population) {

    // Non-owning, 3D cartesian, completing grid (4 dims and sort)
    using grid_t = grid<decltype(ax_n_own), bins::static_array<scalar, 4>,
                        simple_serializer, host_container_types, false>;
    using bin_t = grid_t::bin_type;
    using bin_content_t = std::array<scalar, 4>;

    static_assert(concepts::grid<grid_t>);

    // init
    grid_t::bin_container_type bin_data{};
    bin_data.resize(40'000u, bin_t{});
    // Create non-owning grid
    grid_t g3c(&bin_data, ax_n_own);

    // Test the initialization
    point3 p = {-10.f, -20.f, 0.f};
    bin_t invalid{};
    for (int ib0 = 0; ib0 < 20; ++ib0) {
        for (int ib1 = 0; ib1 < 40; ++ib1) {
            for (int ib2 = 0; ib2 < 100; ib2 += 2) {
                p = {static_cast<scalar>(-10 + ib0),
                     static_cast<scalar>(-20 + ib1),
                     static_cast<scalar>(0 + ib2)};
                test_content(g3c, p, invalid);
            }
        }
    }

    p = {-4.5f, -4.5f, 4.5f};
    bin_content_t expected{4.f, 4.f, 4.f, 4.f};
    // Fill and read
    g3c.template populate<complete<>>(p, 4.f);
    test_content(g3c, p, expected);

    // Test search without search window
    for (scalar entry : g3c.search(p)) {
        EXPECT_EQ(entry, 4.f);
    }

    std::array<dindex, 2> search_window_size{0, 0};

    const auto grid_search1 = g3c.search(p, search_window_size);

    static_assert(detray::ranges::bidirectional_range<decltype(grid_search1)>);

    ASSERT_EQ(grid_search1.size(), 4u);

    for (scalar entry : grid_search1) {
        EXPECT_EQ(entry, 4.f);
    }

    // No neighbors were filled, expect the same candidates
    search_window_size[0] = 1;
    search_window_size[1] = 1;

    const auto grid_search2 = g3c.search(p, search_window_size);

    static_assert(detray::ranges::bidirectional_range<decltype(grid_search2)>);

    ASSERT_EQ(grid_search2.size(), 4u);

    for (scalar entry : grid_search2) {
        EXPECT_EQ(entry, 4.f);
    }

    // Populate some neighboring bins
    point3 p2 = {-5.5f, -4.5f, 4.5f};
    g3c.template populate<complete<>>(p2, 4.f);
    p2 = {-3.5f, -4.5f, 4.5f};
    g3c.template populate<complete<>>(p2, 4.f);
    p2 = {-4.5f, -5.5f, 4.5f};
    g3c.template populate<complete<>>(p2, 4.f);
    p2 = {-4.5f, -3.5f, 4.5f};
    g3c.template populate<complete<>>(p2, 4.f);

    const auto grid_search3 = g3c.search(p, search_window_size);
    ASSERT_EQ(grid_search3.size(), 20u);

    for (scalar entry : grid_search3) {
        EXPECT_EQ(entry, 4.f);
    }
}

/// Test bin entry retrieval
GTEST_TEST(detray_grid, regular_attach_population) {

    // Non-owning, 3D cartesian, completing grid (4 dims and sort)
    using grid_t = grid<decltype(ax_n_own), bins::static_array<scalar, 4>,
                        simple_serializer, host_container_types, false>;
    using bin_t = grid_t::bin_type;
    using bin_content_t = std::array<scalar, 4>;

    static_assert(concepts::grid<grid_t>);

    // init
    grid_t::bin_container_type bin_data{};
    bin_data.resize(40'000u, bin_t{});

    // Create non-owning grid
    grid_t g3ra(&bin_data, ax_n_own);

    // Test the initialization
    point3 p = {-10.f, -20.f, 0.f};
    bin_t invalid{};
    for (int ib0 = 0; ib0 < 20; ++ib0) {
        for (int ib1 = 0; ib1 < 40; ++ib1) {
            for (int ib2 = 0; ib2 < 100; ib2 += 2) {
                p = {static_cast<scalar>(-10 + ib0),
                     static_cast<scalar>(-20 + ib1),
                     static_cast<scalar>(0 + ib2)};
                test_content(g3ra, p, invalid);
            }
        }
    }

    p = {-4.5f, -4.5f, 4.5f};
    bin_content_t expected{5.f, inf, inf, inf};
    // Fill and read
    g3ra.template populate<attach<>>(p, 5.f);
    test_content(g3ra, p, expected);

    // Test search without search window
    for (scalar entry : g3ra.search(p)) {
        EXPECT_EQ(entry, 5.f);
    }

    std::array<dindex, 2> search_window_size{0, 0};

    const auto grid_search1 = g3ra.search(p, search_window_size);

    static_assert(detray::ranges::bidirectional_range<decltype(grid_search1)>);

    ASSERT_EQ(grid_search1.size(), 1u);

    for (scalar entry : grid_search1) {
        EXPECT_EQ(entry, 5.f);
    }

    // No neighbors were filled, expect the same candidates
    search_window_size[0] = 1;
    search_window_size[1] = 1;

    const auto grid_search2 = g3ra.search(p, search_window_size);

    static_assert(detray::ranges::bidirectional_range<decltype(grid_search2)>);

    ASSERT_EQ(grid_search2.size(), 1u);

    for (scalar entry : grid_search2) {
        EXPECT_EQ(entry, 5.f);
    }

    // Put more candidates into bin
    g3ra.template populate<attach<>>(p, 6.f);
    g3ra.template populate<attach<>>(p, 7.f);
    std::vector<scalar> entries{5.f, 6.f, 7.f};

    const auto grid_search3 = g3ra.search(p, search_window_size);
    ASSERT_EQ(grid_search3.size(), 3u);

    for (auto [i, entry] : detray::views::enumerate(grid_search3)) {
        EXPECT_EQ(entry, entries[i]);
    }

    // Populate some neighboring bins
    point3 p2 = {-5.5f, -4.5f, 4.5f};
    g3ra.template populate<attach<>>(p2, 5.f);
    p2 = {-3.5f, -4.5f, 4.5f};
    g3ra.template populate<attach<>>(p2, 5.f);
    p2 = {-4.5f, -5.5f, 4.5f};
    g3ra.template populate<attach<>>(p2, 5.f);
    p2 = {-4.5f, -3.5f, 4.5f};
    g3ra.template populate<attach<>>(p2, 5.f);

    const auto grid_search4 = g3ra.search(p, search_window_size);
    ASSERT_EQ(grid_search4.size(), 7u);

    for (scalar entry : grid_search4) {
        EXPECT_TRUE(entry == 5.f || entry == 6.f || entry == 7.f);
    }
}
