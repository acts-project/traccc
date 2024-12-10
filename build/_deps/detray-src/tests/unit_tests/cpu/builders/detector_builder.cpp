/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/builders/detector_builder.hpp"

#include "detray/builders/cuboid_portal_generator.hpp"
#include "detray/builders/surface_factory.hpp"
#include "detray/core/detector.hpp"
#include "detray/definitions/detail/indexing.hpp"

// Detray test include(s)
#include "detray/test/utils/types.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s)
#include <gtest/gtest.h>

namespace {

using point3 = detray::test::point3;
using vector3 = detray::test::vector3;
using point2 = detray::test::point2;

}  // anonymous namespace

/// Integration test to build a cylinder volume with contained surfaces
GTEST_TEST(detray_builders, detector_builder) {
    using namespace detray;

    using detector_t = detector<>;
    using transform3 = typename detector_t::transform3_type;
    using mask_id = typename detector_t::masks::id;

    // Surface factories
    using trapezoid_factory = surface_factory<detector_t, trapezoid2D>;

    // detector builder
    auto geo_ctx = typename detector_t::geometry_context{};

    detector_builder<default_metadata> det_builder{};

    //
    // first volume builder
    //
    auto vbuilder = det_builder.new_volume(volume_id::e_cylinder);

    typename detector_t::point3_type t{0.f, 0.f, 0.f};
    vbuilder->add_volume_placement(t);

    // initial checks
    EXPECT_EQ(vbuilder->vol_index(), 0u);

    const auto vol_idx{
        static_cast<typename detector_t::surface_type::navigation_link>(
            vbuilder->vol_index())};

    auto trpz_factory = std::make_shared<trapezoid_factory>();
    typename trapezoid_factory::sf_data_collection trpz_sf_data;
    trpz_sf_data.emplace_back(surface_id::e_sensitive,
                              transform3(point3{0.f, 0.f, 1000.f}), vol_idx,
                              std::vector<scalar>{1.f, 3.f, 2.f, 0.25f});
    trpz_sf_data.emplace_back(surface_id::e_sensitive,
                              transform3(point3{0.f, 0.f, 1100.f}), vol_idx,
                              std::vector<scalar>{1.f, 3.f, 2.f, 0.25f});
    trpz_sf_data.emplace_back(surface_id::e_sensitive,
                              transform3(point3{0.f, 0.f, 1200.f}), vol_idx,
                              std::vector<scalar>{1.f, 3.f, 2.f, 0.25f});
    trpz_factory->push_back(std::move(trpz_sf_data));

    vbuilder->add_surfaces(trpz_factory, geo_ctx);

    //
    // second volume builder
    //
    auto vbuilder2 = det_builder.new_volume(volume_id::e_cuboid);

    // volume builder
    t = typename detector_t::point3_type{0.f, 0.f, 20.f};
    vbuilder2->add_volume_placement(t);

    // Add a portal box around the cuboid volume with a min distance of 'env'
    constexpr auto env{0.1f * unit<detray::scalar>::mm};
    auto portal_generator =
        std::make_shared<cuboid_portal_generator<detector_t>>(env);

    vbuilder2->add_surfaces(trpz_factory, geo_ctx);
    vbuilder2->add_surfaces(portal_generator);

    // initial checks
    EXPECT_EQ(vbuilder2->vol_index(), 1u);

    //
    // build the detector
    //
    vecmem::host_memory_resource host_mr;
    const detector_t d = det_builder.build(host_mr);
    const auto& vol0 = tracking_volume{d, 0u};
    const auto& vol1 = tracking_volume{d, 1u};

    // check the results
    EXPECT_EQ(d.volumes().size(), 2u);
    EXPECT_EQ(vol0.id(), volume_id::e_cylinder);
    EXPECT_EQ(vol0.index(), 0u);
    EXPECT_EQ(vol1.id(), volume_id::e_cuboid);
    EXPECT_EQ(vol1.index(), 1u);

    // Check the volume placements for both volumes
    typename detector_t::transform3_type identity{};
    EXPECT_TRUE(vol0.transform() == identity);
    EXPECT_TRUE(d.transform_store().at(0u) == identity);
    typename detector_t::transform3_type trf{t};
    EXPECT_TRUE(vol1.transform() == trf);
    EXPECT_TRUE(d.transform_store().at(4u) == trf);

    EXPECT_EQ(d.surfaces().size(), 12u);
    EXPECT_EQ(d.mask_store().template size<mask_id::e_portal_cylinder2>(), 0u);
    EXPECT_EQ(d.mask_store().template size<mask_id::e_portal_ring2>(), 0u);
    EXPECT_EQ(d.mask_store().template size<mask_id::e_annulus2>(), 0u);
    EXPECT_EQ(d.mask_store().template size<mask_id::e_cylinder2>(), 0u);
    EXPECT_EQ(d.mask_store().template size<mask_id::e_rectangle2>(), 3u);
    EXPECT_EQ(d.mask_store().template size<mask_id::e_ring2>(), 0u);
    EXPECT_EQ(d.mask_store().template size<mask_id::e_trapezoid2>(), 6u);
}
