/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// TODO: Remove this when gcc fixes their false positives.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic warning "-Warray-bounds"
#endif

// Project include(s)
#include "detray/builders/cuboid_portal_generator.hpp"
#include "detray/builders/surface_factory.hpp"
#include "detray/builders/volume_builder.hpp"
#include "detray/core/detector.hpp"
#include "detray/definitions/geometry.hpp"
#include "detray/definitions/units.hpp"
#include "detray/io/frontend/detector_writer.hpp"
#include "detray/test/utils/detectors/build_toy_detector.hpp"

// Example include(s)
#include "detray/tutorial/detector_metadata.hpp"
#include "detray/tutorial/square_surface_generator.hpp"
#include "detray/tutorial/types.hpp"  // linear algebra types

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s)
#include <limits>
#include <memory>

/// Write a dector using the json IO
int main() {

    // The new detector type
    using detector_t = detray::detector<detray::tutorial::my_metadata>;

    // First, create an empty detector in in host memory to be filled
    vecmem::host_memory_resource host_mr;
    detector_t det{host_mr};

    // Now fill the detector

    // Get a generic volume builder first and decorate it later
    detray::volume_builder<detector_t> vbuilder{detray::volume_id::e_cuboid};

    // Fill some squares into the volume
    using square_factory_t =
        detray::surface_factory<detector_t, detray::tutorial::square2D>;
    auto sq_factory = std::make_shared<square_factory_t>();

    // Add a square that is 20x20mm large, links back to its mother volume (0)
    // and is placed with a translation of (x = 1mm, y = 2mm, z = 3mm)
    detray::tutorial::vector3 translation{
        1.f * detray::unit<detray::scalar>::mm,
        2.f * detray::unit<detray::scalar>::mm,
        3.f * detray::unit<detray::scalar>::mm};
    sq_factory->push_back({detray::surface_id::e_sensitive,
                           detray::tutorial::transform3{translation},
                           0u,
                           {20.f * detray::unit<detray::scalar>::mm}});

    // Add some programmatically generated square surfaces
    auto sq_generator =
        std::make_shared<detray::tutorial::square_surface_generator>(
            10, 10.f * detray::unit<detray::scalar>::mm);

    // Add a portal box around the cuboid volume with a min distance of 'env'
    constexpr auto env{0.1f * detray::unit<detray::scalar>::mm};
    auto portal_generator =
        std::make_shared<detray::cuboid_portal_generator<detector_t>>(env);

    // Add surfaces to volume and add the volume to the detector
    vbuilder.add_surfaces(sq_factory);
    vbuilder.add_surfaces(sq_generator);
    vbuilder.add_surfaces(portal_generator);

    vbuilder.build(det);

    // Write the detector to file
    auto writer_cfg = detray::io::detector_writer_config{}
                          .format(detray::io::format::json)
                          .replace_files(true);
    detray::io::write_detector(det, {{0u, "example_detector"}}, writer_cfg);
}
