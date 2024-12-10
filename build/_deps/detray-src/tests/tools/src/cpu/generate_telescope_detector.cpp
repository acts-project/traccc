/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/definitions/units.hpp"
#include "detray/geometry/mask.hpp"
#include "detray/geometry/shapes.hpp"

// Detray IO include(s)
#include "detray/io/frontend/detector_writer.hpp"

// Detray test include(s)
#include "detray/options/detector_io_options.hpp"
#include "detray/options/parse_options.hpp"
#include "detray/test/utils/detectors/build_telescope_detector.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// Boost
#include "detray/options/boost_program_options.hpp"

namespace po = boost::program_options;

using namespace detray;

namespace {

/// Generate and write a telescope detector, given the commandline variables
/// and a configuration for the detector writer @param writer_cfg
template <typename mask_shape_t, typename value_t, typename trajectory_t>
void write_telecope(const po::variables_map &vm,
                    io::detector_writer_config &writer_cfg,
                    std::vector<value_t> &mask_params,
                    const trajectory_t &traj) {

    using detector_t = detector<telescope_metadata<mask_shape_t>>;
    using scalar_t = typename detector_t::scalar_type;

    detray::tel_det_config<mask_shape_t, trajectory_t> tel_cfg{mask_params,
                                                               traj};

    tel_cfg.n_surfaces(vm["modules"].as<unsigned int>());
    tel_cfg.length(vm["length"].as<scalar_t>());
    tel_cfg.mat_thickness(vm["thickness"].as<scalar_t>());
    tel_cfg.envelope(vm["envelope"].as<scalar_t>());

    // Build the detector
    vecmem::host_memory_resource host_mr;
    auto [tel_det, tel_names] = build_telescope_detector(host_mr, tel_cfg);

    // Write to file
    detray::io::write_detector(tel_det, tel_names, writer_cfg);
}

/// Generate and write a telescope detector, given the commandline variables
/// and a configuration for the detector writer @param writer_cfg
template <typename mask_shape_t, typename value_t>
void write_telecope(const po::variables_map &vm,
                    io::detector_writer_config &writer_cfg,
                    std::vector<value_t> &mask_params) {

    using detector_t = detector<telescope_metadata<mask_shape_t>>;
    using scalar_t = typename detector_t::scalar_type;
    using algebra_t = typename detector_t::algebra_type;
    using vector3_t = dvector3D<algebra_t>;

    // Construct the pilot track
    std::string direction{vm["direction"].as<std::string>()};
    vector3_t dir;
    if (direction == "x") {
        dir = {1.f, 0.f, 0.f};
    } else if (direction == "y") {
        dir = {{0.f, 1.f, 0.f}};
    } else if (direction == "z") {
        dir = {0.f, 0.f, 1.f};
    }

    if (!vm.count("b_field")) {
        detray::detail::ray<algebra_t> r{};
        r.set_dir(dir);

        write_telecope<mask_shape_t>(vm, writer_cfg, mask_params, r);
    } else {
        const auto b_field = vm["b_field"].as<std::vector<scalar_t>>();
        if (b_field.size() == 3u) {
            vector3_t B{b_field[0] * unit<scalar_t>::T,
                        b_field[1] * unit<scalar_t>::T,
                        b_field[2] * unit<scalar_t>::T};

            const auto p{vm["p"].as<scalar_t>() * unit<scalar_t>::GeV};
            detray::detail::helix<algebra_t> h{
                {0.f, 0.f, 0.f}, 0.f, dir, 1.f / p, &B};

            write_telecope<mask_shape_t>(vm, writer_cfg, mask_params, h);
        } else {
            throw std::invalid_argument(
                "B-field vector has to have three entries");
        }
    }
}

}  // anonymous namespace

int main(int argc, char **argv) {

    using scalar_t = detray::scalar;

    // Options parsing
    po::options_description desc("\nTelescope detector generation options");

    std::vector<scalar_t> mask_params{
        20.f * unit<scalar_t>::mm,
        20.f * unit<scalar_t>::mm};  // < default values for rectangles
    desc.add_options()("modules", po::value<unsigned int>()->default_value(10u),
                       "number of modules in telescope [1-20]")(
        "type", po::value<std::string>()->default_value("rectangle"),
        "type of the telescope modules [rectangle, trapezoid, annulus, ring, "
        "cylinder]")(
        "params", po::value<std::vector<scalar_t>>(&mask_params)->multitoken(),
        "Mask values for the shape given in 'type'")(
        "length",
        po::value<scalar_t>()->default_value(500.f * unit<scalar_t>::mm),
        "length of the telescope [mm]")(
        "thickness",
        po::value<scalar_t>()->default_value(1.f * unit<scalar_t>::mm),
        "thickness of the module silicon material")(
        "envelope",
        po::value<scalar_t>()->default_value(100.f * unit<scalar_t>::um),
        "minimal distance between sensitive surfaces and portals")(
        "direction", po::value<std::string>()->default_value("z"),
        "direction of the telescope in global frame [x, y, z]")(
        "b_field",
        boost::program_options::value<std::vector<scalar_t>>()->multitoken(),
        "B field vector for a pilot helix [T]")(
        "p", po::value<scalar_t>()->default_value(10.f * unit<scalar_t>::GeV),
        "Total momentum of the pilot track [GeV]");

    // Configuration
    detray::io::detector_writer_config writer_cfg{};
    writer_cfg.format(detray::io::format::json).replace_files(false);
    // Default output path
    writer_cfg.path("./telescope_detector/");

    // Parse options
    po::variables_map vm =
        detray::options::parse_options(desc, argc, argv, writer_cfg);

    // Build the geometry
    std::string type{vm["type"].as<std::string>()};
    if (type == "rectangle") {
        write_telecope<rectangle2D>(vm, writer_cfg, mask_params);
    } else if (type == "trapezoid") {
        write_telecope<trapezoid2D>(vm, writer_cfg, mask_params);
    } else if (type == "annulus") {
        write_telecope<annulus2D>(vm, writer_cfg, mask_params);
    } else if (type == "ring") {
        write_telecope<ring2D>(vm, writer_cfg, mask_params);
    } else if (type == "cylinder") {
        write_telecope<cylinder2D>(vm, writer_cfg, mask_params);
    }
}
