/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/definitions/units.hpp"
#include "detray/geometry/mask.hpp"
#include "detray/materials/predefined_materials.hpp"
#include "detray/navigation/detail/trajectories.hpp"
#include "detray/navigation/volume_graph.hpp"
#include "detray/test/utils/detectors/build_telescope_detector.hpp"
#include "detray/test/utils/detectors/build_toy_detector.hpp"
#include "detray/tracks/tracks.hpp"

// Example linear algebra plugin: std::array
#include "detray/tutorial/types.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s)
#include <cmath>
#include <cstdlib>
#include <iostream>

/// Show how to build the test detectors.
///
/// Toy detector: Pixel section of the ACTS Generic detector (TrackML).
///
/// Telescope detector: Array of surfaces of a given type in a bounding portal
///                     box.
int main(int argc, char** argv) {

    // Memory resource to allocate the detector data stores
    vecmem::host_memory_resource host_mr;

    //
    // Toy detector
    //
    detray::toy_det_config toy_cfg{};
    // Number of barrel layers (0 - 4)
    toy_cfg.n_brl_layers(4u);
    // Number of endcap layers on either side (0 - 7)
    // Note: The detector must be configured with 4 barrel layers to be able to
    // add any encap layers
    toy_cfg.n_edc_layers(1u);

    // Read toy detector config from commandline, if it was given
    if (argc == 3) {
        toy_cfg.n_brl_layers(
            static_cast<unsigned int>(std::abs(std::atoi(argv[1]))));
        toy_cfg.n_edc_layers(
            static_cast<unsigned int>(std::abs(std::atoi(argv[2]))));
    }

    // Fill the detector
    const auto [toy_det, names] = detray::build_toy_detector(host_mr, toy_cfg);

    // Print the volume graph of the toy detector
    std::cout << "\nToy detector:\n"
              << "-------------\n"
              << detray::volume_graph{toy_det}.to_string() << std::endl;

    //
    // Telescope detector
    //

    // The telescope detector is built according to a 'pilot trajectory'
    // (either a helix or a ray) along which the modules are placed in given
    // distances. The world portals are constructed from a bounding box around
    // the test surfaces (the envelope is configurable).

    // Mask with a rectangular shape (20x20 mm)

    detray::mask<detray::rectangle2D> rectangle{
        0u, 20.f * detray::unit<detray::scalar>::mm,
        20.f * detray::unit<detray::scalar>::mm};

    // Mask with a trapezoid shape
    using trapezoid_t = detray::mask<detray::trapezoid2D>;

    constexpr detray::scalar hx_min_y{10.f * detray::unit<detray::scalar>::mm};
    constexpr detray::scalar hx_max_y{30.f * detray::unit<detray::scalar>::mm};
    constexpr detray::scalar hy{20.f * detray::unit<detray::scalar>::mm};
    constexpr detray::scalar divisor{10.f / (20.f * hy)};
    trapezoid_t trapezoid{0u, hx_min_y, hx_max_y, hy, divisor};

    // Build from given module positions (places 11 surfaces)
    std::vector<detray::scalar> positions = {
        0.f * detray::unit<detray::scalar>::mm,
        50.f * detray::unit<detray::scalar>::mm,
        100.f * detray::unit<detray::scalar>::mm,
        150.f * detray::unit<detray::scalar>::mm,
        200.f * detray::unit<detray::scalar>::mm,
        250.f * detray::unit<detray::scalar>::mm,
        300.f * detray::unit<detray::scalar>::mm,
        350.f * detray::unit<detray::scalar>::mm,
        400.f * detray::unit<detray::scalar>::mm,
        450.f * detray::unit<detray::scalar>::mm,
        500.f * detray::unit<detray::scalar>::mm};

    //
    // Case 1: Defaults: Straight telescope in z-direction,
    //         10 rectangle surfaces, 500mm in length, modules evenly spaced,
    //         silicon material (80mm)
    const auto [tel_det1, tel_names1] =
        detray::build_telescope_detector<detray::rectangle2D>(host_mr);

    std::cout << "\nTelescope detector - case 1:\n"
              << "----------------------------\n"
              << detray::volume_graph{tel_det1}.to_string() << std::endl;

    //
    // Case 2: Straight telescope in z-direction, 15 trapezoid surfaces, 2000mm
    //         in length, modules evenly spaced, silicon material (80mm)
    detray::tel_det_config trp_cfg{trapezoid};
    trp_cfg.n_surfaces(15).length(2000.f * detray::unit<detray::scalar>::mm);

    const auto [tel_det2, tel_names2] =
        detray::build_telescope_detector(host_mr, trp_cfg);

    std::cout << "\nTelescope detector - case 2:\n"
              << "----------------------------\n"
              << detray::volume_graph{tel_det2}.to_string() << std::endl;

    //
    // Case 3: Straight telescope in x-direction, 11 rectangle surfaces, 2000mm
    //         in length, modules places according to 'positions',
    //         silicon material (80mm)

    // Pilot trajectory in x-direction
    detray::detail::ray<detray::tutorial::algebra_t> x_track{
        {0.f, 0.f, 0.f}, 0.f, {1.f, 0.f, 0.f}, -1.f};

    detray::tel_det_config rct_cfg{rectangle};
    rct_cfg.positions(positions).pilot_track(x_track);

    const auto [tel_det3, tel_names3] =
        build_telescope_detector(host_mr, rct_cfg);

    std::cout << "\nTelescope detector - case 3:\n"
              << "----------------------------\n"
              << detray::volume_graph{tel_det3}.to_string() << std::endl;

    //
    // Case 4: Bent telescope along helical track, 11 trapezoid surfaces,
    //         modules spaced according to given positions,
    //         silicon material (80mm)

    // Pilot track in x-direction
    detray::free_track_parameters<detray::tutorial::algebra_t> y_track{
        {0.f, 0.f, 0.f}, 0.f, {1.f, 0.f, 0.f}, -1.f};

    // Helix in a constant B-field 1T in z-direction
    using helix_t = detray::detail::helix<detray::tutorial::algebra_t>;
    detray::tutorial::vector3 B_z{0.f, 0.f,
                                  1.f * detray::unit<detray::scalar>::T};
    helix_t helix(y_track, &B_z);

    detray::tel_det_config htrp_cfg{trapezoid, helix};
    htrp_cfg.positions(positions);

    const auto [tel_det4, tel_names4] =
        detray::build_telescope_detector(host_mr, htrp_cfg);

    std::cout << "\nTelescope detector - case 4:\n"
              << "----------------------------\n"
              << detray::volume_graph{tel_det4}.to_string() << std::endl;
}
