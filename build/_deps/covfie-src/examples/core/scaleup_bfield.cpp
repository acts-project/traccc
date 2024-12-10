/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <fstream>
#include <iostream>

#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>

#include <covfie/core/algebra/affine.hpp>
#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/nearest_neighbour.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/parameter_pack.hpp>

void parse_opts(
    int argc, char * argv[], boost::program_options::variables_map & vm
)
{
    boost::program_options::options_description opts("general options");

    opts.add_options()("help", "produce help message")(
        "input,i",
        boost::program_options::value<std::string>()->required(),
        "input magnetic field to read"
    )("output,o",
      boost::program_options::value<std::string>()->required(),
      "output magnetic field to write");

    boost::program_options::parsed_options parsed =
        boost::program_options::command_line_parser(argc, argv)
            .options(opts)
            .run();

    boost::program_options::store(parsed, vm);

    if (vm.count("help")) {
        std::cout << opts << std::endl;
        std::exit(0);
    }

    try {
        boost::program_options::notify(vm);
    } catch (boost::program_options::required_option & e) {
        BOOST_LOG_TRIVIAL(error) << e.what();
        std::exit(1);
    }
}

using inner_backend_t = covfie::backend::strided<
    covfie::vector::size3,
    covfie::backend::array<covfie::vector::float3>>;

using input_field_t = covfie::field<
    covfie::backend::affine<covfie::backend::linear<inner_backend_t>>>;

using output_field_t = covfie::field<covfie::backend::affine<
    covfie::backend::nearest_neighbour<inner_backend_t>>>;

int main(int argc, char ** argv)
{
    boost::program_options::variables_map vm;
    parse_opts(argc, argv, vm);

    BOOST_LOG_TRIVIAL(info) << "Welcome to the covfie magnetic field renderer!";
    BOOST_LOG_TRIVIAL(info) << "Using magnetic field file \""
                            << vm["input"].as<std::string>() << "\"";
    BOOST_LOG_TRIVIAL(info) << "Starting read of input file...";

    std::ifstream ifs(vm["input"].as<std::string>(), std::ifstream::binary);

    if (!ifs.good()) {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to open input file "
                                 << vm["input"].as<std::string>() << "!";
        std::exit(1);
    }

    input_field_t old_field(ifs);
    ifs.close();

    BOOST_LOG_TRIVIAL(info) << "Creating a new, larger field...";

    covfie::algebra::affine<3> translation =
        covfie::algebra::affine<3>::translation(10000.f, 10000.f, 15000.f);
    covfie::algebra::affine<3> scaling = covfie::algebra::affine<3>::scaling(
        600 / 20000.f, 600 / 20000.f, 900 / 30000.f
    );

    output_field_t new_field(covfie::make_parameter_pack(
        scaling * translation,
        std::monostate{},
        covfie::utility::nd_size<3>{601u, 601u, 901u}
    ));

    BOOST_LOG_TRIVIAL(info) << "Copying data from old field to new field...";

    input_field_t::view_t iv(old_field);
    output_field_t::view_t ov(new_field);

    for (std::size_t x = 0; x < 601; ++x) {
        for (std::size_t y = 0; y < 601; ++y) {
            for (std::size_t z = 0; z < 901; ++z) {
                ov.at(
                    -10000.f + static_cast<float>(x) * 33.333333333f,
                    -10000.f + static_cast<float>(y) * 33.333333333f,
                    -15000.f + static_cast<float>(z) * 33.333333333f
                ) =
                    iv.at(
                        -10000.f + static_cast<float>(x) * 33.333333333f,
                        -10000.f + static_cast<float>(y) * 33.333333333f,
                        -15000.f + static_cast<float>(z) * 33.333333333f
                    );
            }
        }
    }

    BOOST_LOG_TRIVIAL(info) << "Writing magnetic field to file \""
                            << vm["output"].as<std::string>() << "\"...";

    std::ofstream fs(vm["output"].as<std::string>(), std::ofstream::binary);

    if (!fs.good()) {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to open output file "
                                 << vm["output"].as<std::string>() << "!";
        std::exit(1);
    }

    new_field.dump(fs);

    fs.close();

    BOOST_LOG_TRIVIAL(info) << "Scaling complete, goodbye!";

    return 0;
}
