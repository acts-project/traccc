/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/io/digitization_configurator.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/write_digitization_config.hpp"
#include "traccc/options/handle_argument_errors.hpp"

// Detray include(s)
#include "detray/core/detector.hpp"
#include "detray/geometry/surface.hpp"
#include "detray/io/common/detector_reader.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// Boost
#include <boost/program_options.hpp>

// System include(s)
#include <sstream>
#include <stdexcept>
#include <string>

namespace po = boost::program_options;

/// @brief create a detailed digitization file for the given detector
///
/// Takes a human editable digitization "meta"-configuration file and assigns
/// the given configuration to the specified surfaces by visitng the geometry.
/// For every discovered surface in a digitization domain, the respective
/// digitization configuration is written into the output json file.
///
/// @param digi_meta_config_file file that contains the editable "meta"-config
/// @param reader_cfg detray detector reader config, specifying the geometry
int create_digitization_config(
    const std::string& digi_meta_config_file,
    const detray::io::detector_reader_config& reader_cfg) {

    vecmem::host_memory_resource host_mr;

    const auto [det, names] =
        detray::io::read_detector<detray::detector<>>(host_mr, reader_cfg);

    std::cout << "Running digitization configurator for \"" << names.at(0)
              << "\"..." << std::endl;

    const traccc::digitization_config digi_meta_cfg =
        traccc::io::read_digitization_config(digi_meta_config_file);

    traccc::io::digitization_configurator digi_configurator{digi_meta_cfg};

    // Visit all detector surfaces to generate their digitization configuration
    for (const auto& sf_desc : det.surface_lookup()) {
        digi_configurator(detray::surface{det, sf_desc});
    }

    std::string outfile_name = names.at(0) + "-geometric-config.json";
    traccc::io::write_digitization_config(outfile_name,
                                          digi_configurator.output_digi_cfg);

    std::cout << "Done. Written config to: " << outfile_name << std::endl;

    return EXIT_SUCCESS;
}

// The main routine
//
int main(int argc, char* argv[]) {
    // Options parsing
    po::options_description desc("\ntraccc digitization configurator");

    desc.add_options()("help", "produce help message")(
        "geometry_file", po::value<std::string>(),
        "detray geometry input file")("digi_meta_config_file",
                                      po::value<std::string>(),
                                      "digitization meta-configuration file");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    traccc::handle_argument_errors(vm, desc);

    // Configs to be filled
    detray::io::detector_reader_config reader_cfg{};

    // Input files
    if (vm.count("geometry_file")) {
        reader_cfg.add_file(vm["geometry_file"].as<std::string>());
    } else {
        std::stringstream err_stream{};
        err_stream << "Please specify a geometry input file!\n\n" << desc;

        throw std::invalid_argument(err_stream.str());
    }
    std::string digi_meta_config_file;
    if (vm.count("digi_meta_config_file")) {
        digi_meta_config_file = vm["digi_meta_config_file"].as<std::string>();
    } else {
        std::stringstream err_stream{};
        err_stream << "Please specify a digitization meta-configuration input"
                   << " file!\n\n"
                   << desc;

        throw std::invalid_argument(err_stream.str());
    }

    return create_digitization_config(digi_meta_config_file, reader_cfg);
}
