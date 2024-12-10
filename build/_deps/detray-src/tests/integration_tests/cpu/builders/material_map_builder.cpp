/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Detray include(s)
#include "detray/builders/material_map_builder.hpp"

#include "detray/builders/cuboid_portal_generator.hpp"
#include "detray/builders/detector_builder.hpp"
#include "detray/builders/material_map_factory.hpp"
#include "detray/builders/surface_factory.hpp"
#include "detray/builders/volume_builder.hpp"
#include "detray/core/detector.hpp"
#include "detray/definitions/detail/indexing.hpp"
#include "detray/utils/ranges.hpp"

// Detray test include(s)
#include "detray/test/utils/types.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s)
#include <gtest/gtest.h>

// System include(s)
#include <limits>
#include <memory>
#include <vector>

using namespace detray;

namespace {

using point3 = test::point3;

using detector_t = detector<>;
using mat_id = typename detector_t::materials::id;
using bin_index_t = axis::multi_bin<2u>;

/// Add generate input material for material maps
template <typename material_factory_t, typename scalar_t>
auto add_material_data(const material_factory_t& mat_factory, mat_id id,
                       std::size_t sf_index, scalar_t t,
                       material<scalar_t> mat = silicon<scalar_t>()) {

    typename material_factory_t::element_type::data_type mat_data{sf_index};
    std::vector<bin_index_t> m_bins{};
    std::vector<std::size_t> n_bins{5u, 10u};
    std::vector<std::vector<scalar_t>> axis_spans = {};

    // Add material for every bin
    for (auto [i, j] : detray::views::cartesian_product{
             detray::views::iota{0u, 5u}, detray::views::iota{0u, 10u}}) {
        m_bins.push_back({i, j});
        mat_data.append(t, mat);
        t += 0.25f * unit<scalar_t>::mm;
    }

    mat_factory->add_material(id, std::move(mat_data), std::move(n_bins),
                              std::move(axis_spans), std::move(m_bins));
}

}  // anonymous namespace

/// Integration test: material builder as volume builder decorator
GTEST_TEST(detray_builders, decorator_material_map_builder) {

    using transform3 = typename detector_t::transform3_type;
    using scalar_t = typename detector_t::scalar_type;
    using mask_id = typename detector_t::masks::id;

    using pt_cylinder_factory_t =
        surface_factory<detector_t, concentric_cylinder2D>;
    using rectangle_factory = surface_factory<detector_t, rectangle2D>;
    using trapezoid_factory = surface_factory<detector_t, trapezoid2D>;
    using cylinder_factory = surface_factory<detector_t, cylinder2D>;

    using mat_factory_t = material_map_factory<detector_t, bin_index_t>;

    vecmem::host_memory_resource host_mr;
    detector_t d(host_mr);
    auto geo_ctx = typename detector_t::geometry_context{};

    auto vbuilder =
        std::make_unique<volume_builder<detector_t>>(volume_id::e_cylinder);
    auto mat_builder = material_map_builder<detector_t>{std::move(vbuilder)};

    EXPECT_TRUE(d.volumes().size() == 0);

    // Add some portals first
    auto pt_cyl_factory = std::make_unique<pt_cylinder_factory_t>();

    pt_cyl_factory->push_back({surface_id::e_portal,
                               transform3(point3{0.f, 0.f, 0.f}), 1u,
                               std::vector<scalar>{10.f, -1500.f, 1500.f}});
    pt_cyl_factory->push_back({surface_id::e_portal,
                               transform3(point3{0.f, 0.f, 0.f}), 2u,
                               std::vector<scalar>{20.f, -1500.f, 1500.f}});

    // Then some passive and sensitive surfaces
    auto rect_factory = std::make_unique<rectangle_factory>();

    typename rectangle_factory::sf_data_collection rect_sf_data;
    rect_sf_data.emplace_back(surface_id::e_sensitive,
                              transform3(point3{0.f, 0.f, -10.f}), 0u,
                              std::vector<scalar>{10.f, 8.f});
    rect_sf_data.emplace_back(surface_id::e_sensitive,
                              transform3(point3{0.f, 0.f, -20.f}), 0u,
                              std::vector<scalar>{10.f, 8.f});
    rect_sf_data.emplace_back(surface_id::e_sensitive,
                              transform3(point3{0.f, 0.f, -30.f}), 0u,
                              std::vector<scalar>{10.f, 8.f});
    rect_factory->push_back(std::move(rect_sf_data));

    auto trpz_factory = std::make_unique<trapezoid_factory>();

    trpz_factory->push_back({surface_id::e_sensitive,
                             transform3(point3{0.f, 0.f, 1000.f}), 0u,
                             std::vector<scalar>{1.f, 3.f, 2.f, 0.25f}});

    auto cyl_factory = std::make_unique<cylinder_factory>();

    cyl_factory->push_back({surface_id::e_passive,
                            transform3(point3{0.f, 0.f, 0.f}), 0u,
                            std::vector<scalar>{5.f, -1300.f, 1300.f}});

    // Now add the material for each surface
    auto mat_pt_cyl_factory =
        std::make_shared<mat_factory_t>(std::move(pt_cyl_factory));
    scalar_t t{1.f * unit<scalar_t>::mm};
    add_material_data(mat_pt_cyl_factory, mat_id::e_concentric_cylinder2_map,
                      0u, t, silicon<scalar_t>());
    add_material_data(mat_pt_cyl_factory, mat_id::e_concentric_cylinder2_map,
                      1u, t, silicon<scalar_t>());

    auto mat_rect_factory =
        std::make_shared<mat_factory_t>(std::move(rect_factory));
    t = 1.f * unit<scalar_t>::mm;
    add_material_data(mat_rect_factory, mat_id::e_rectangle2_map, 2u, t,
                      tungsten<scalar_t>());
    // No material for surface with index 3
    t = 3.f * unit<scalar_t>::mm;
    add_material_data(mat_rect_factory, mat_id::e_rectangle2_map, 4u, t,
                      tungsten<scalar_t>());

    auto mat_trpz_factory =
        std::make_shared<mat_factory_t>(std::move(trpz_factory));
    t = 1.f * unit<scalar_t>::mm;
    add_material_data(mat_trpz_factory, mat_id::e_trapezoid2_map, 5u, t,
                      tungsten<scalar_t>());

    auto mat_cyl_factory =
        std::make_shared<mat_factory_t>(std::move(cyl_factory));
    t = 1.5f * unit<scalar_t>::mm;
    add_material_data(mat_cyl_factory, mat_id::e_cylinder2_map, 6u, t,
                      gold<scalar_t>());

    // Add surfaces and material to detector
    mat_builder.add_surfaces(mat_pt_cyl_factory, geo_ctx);
    mat_builder.add_surfaces(mat_rect_factory, geo_ctx);
    mat_builder.add_surfaces(mat_trpz_factory, geo_ctx);
    mat_builder.add_surfaces(mat_cyl_factory, geo_ctx);

    // Add the volume to the detector
    mat_builder.build(d);

    //
    // check results
    //
    const auto& vol = d.volumes().back();
    EXPECT_TRUE(d.volumes().size() == 1u);
    EXPECT_EQ(vol.index(), 0u);
    EXPECT_EQ(vol.id(), volume_id::e_cylinder);

    EXPECT_EQ(d.surfaces().size(), 7u);
    EXPECT_EQ(d.transform_store().size(), 8u);
    EXPECT_EQ(d.mask_store().template size<mask_id::e_portal_cylinder2>(), 2u);
    EXPECT_EQ(d.mask_store().template size<mask_id::e_portal_ring2>(), 0u);
    EXPECT_EQ(d.mask_store().template size<mask_id::e_cylinder2>(), 1u);
    EXPECT_EQ(d.mask_store().template size<mask_id::e_rectangle2>(), 3u);
    EXPECT_EQ(d.mask_store().template size<mask_id::e_trapezoid2>(), 1u);

    EXPECT_EQ(d.material_store().template size<mat_id::e_slab>(), 0u);
    EXPECT_EQ(d.material_store().template size<mat_id::e_rod>(), 0u);
    EXPECT_EQ(d.material_store().template size<mat_id::e_disc2_map>(), 0u);
    EXPECT_EQ(d.material_store().template size<mat_id::e_annulus2_map>(), 0u);
    EXPECT_EQ(d.material_store().template size<mat_id::e_drift_cell_map>(), 0u);
    EXPECT_EQ(d.material_store().template size<mat_id::e_straw_tube_map>(), 0u);
    EXPECT_EQ(d.material_store().template size<mat_id::e_cylinder2_map>(), 1u);
    EXPECT_EQ(
        d.material_store().template size<mat_id::e_concentric_cylinder2_map>(),
        2u);
    // Rectangle and trapezoid surfaces have the same grid geometry
    EXPECT_EQ(d.material_store().template size<mat_id::e_rectangle2_map>(), 3u);
    EXPECT_EQ(d.material_store().template size<mat_id::e_trapezoid2_map>(), 3u);

    // Check the material links
    std::size_t pt_cyl_idx{0u};
    std::size_t cyl_idx{0u};
    std::size_t cart_idx{0u};
    for (auto& sf_desc : d.surfaces()) {
        const auto& mat_link = sf_desc.material();
        switch (mat_link.id()) {
            case mat_id::e_cylinder2_map: {
                EXPECT_EQ(mat_link.index(), cyl_idx++) << sf_desc;
                break;
            }
            case mat_id::e_concentric_cylinder2_map: {
                EXPECT_EQ(mat_link.index(), pt_cyl_idx++) << sf_desc;
                break;
            }
            case mat_id::e_rectangle2_map: {
                EXPECT_EQ(mat_link.index(), cart_idx++) << sf_desc;
                break;
            }
            case mat_id::e_none: {
                // No material on surface 3
                EXPECT_TRUE(sf_desc.index() == 3u)
                    << "No material on: " << sf_desc;
                EXPECT_TRUE(mat_link.id() == mat_id::e_none) << sf_desc;
                EXPECT_TRUE(mat_link.index() == 0) << sf_desc;
                break;
            }
            default: {
                EXPECT_TRUE(false) << sf_desc;
            }
        }
    }

    // Check the material map content
    for (auto cyl_mat_grid :
         d.material_store()
             .template get<mat_id::e_concentric_cylinder2_map>()) {

        EXPECT_EQ(cyl_mat_grid.nbins(), 50u);
        EXPECT_EQ(cyl_mat_grid.size(), 50u);

        auto r_axis = cyl_mat_grid.template get_axis<axis::label::e_rphi>();
        EXPECT_EQ(r_axis.nbins(), 5u);
        auto z_axis = cyl_mat_grid.template get_axis<axis::label::e_cyl_z>();
        EXPECT_EQ(z_axis.nbins(), 10u);

        for (const auto& mat_slab : cyl_mat_grid.all()) {
            EXPECT_TRUE(mat_slab.get_material() == silicon<scalar>() ||
                        mat_slab.get_material() == gold<scalar>());
        }
    }

    for (auto cart_mat_grid :
         d.material_store().template get<mat_id::e_rectangle2_map>()) {

        EXPECT_EQ(cart_mat_grid.nbins(), 50u);
        EXPECT_EQ(cart_mat_grid.size(), 50u);

        auto x_axis = cart_mat_grid.template get_axis<axis::label::e_x>();
        EXPECT_EQ(x_axis.nbins(), 5u);
        auto y_axis = cart_mat_grid.template get_axis<axis::label::e_y>();
        EXPECT_EQ(y_axis.nbins(), 10u);

        for (const auto& mat_slab : cart_mat_grid.all()) {
            EXPECT_TRUE(mat_slab.get_material() == tungsten<scalar>());
        }
    }
}
