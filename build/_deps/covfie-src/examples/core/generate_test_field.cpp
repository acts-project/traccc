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
        "output,o",
        boost::program_options::value<std::string>()->required(),
        "output magnetic field to write"
    );

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

using core_backend_t = covfie::backend::strided<
    covfie::vector::size3,
    covfie::backend::array<covfie::vector::float3>>;

using core_field_t = covfie::field<core_backend_t>;

using full_backend_t =
    covfie::backend::affine<covfie::backend::nearest_neighbour<core_backend_t>>;

using full_field_t = covfie::field<full_backend_t>;

int main(int argc, char ** argv)
{
    boost::program_options::variables_map vm;
    parse_opts(argc, argv, vm);

    BOOST_LOG_TRIVIAL(info) << "Welcome to the vector field generator!";
    BOOST_LOG_TRIVIAL(info) << "Allocating space for new vector field...";

    core_field_t cf(covfie::make_parameter_pack(core_backend_t::configuration_t{
        201u, 201u, 301u}));

    BOOST_LOG_TRIVIAL(info) << "Filling new vector field...";

    core_field_t::view_t cfv(cf);

    for (std::size_t x = 0; x < 201; ++x) {
        for (std::size_t y = 0; y < 201; ++y) {
            for (std::size_t z = 0; z < 301; ++z) {
                cfv.at(x, y, z) = {0.f, 0.f, 2.f};
            }
        }
    }

    BOOST_LOG_TRIVIAL(info) << "Creating affine-transformed field...";

    covfie::algebra::affine<3> translation =
        covfie::algebra::affine<3>::translation(10000.f, 10000.f, 15000.f);
    covfie::algebra::affine<3> scaling = covfie::algebra::affine<3>::scaling(
        200 / (20000.f), 200 / (20000.f), 300 / (30000.f)
    );

    full_field_t ff(covfie::make_parameter_pack(
        full_backend_t::configuration_t(scaling * translation),
        full_backend_t::backend_t::configuration_t{},
        cf.backend()
    ));

    BOOST_LOG_TRIVIAL(info) << "Writing final field to file \""
                            << vm["output"].as<std::string>() << "\"...";

    std::ofstream fs(vm["output"].as<std::string>(), std::ofstream::binary);

    if (!fs.good()) {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to open output file "
                                 << vm["output"].as<std::string>() << "!";
        std::exit(1);
    }

    ff.dump(fs);
    fs.close();

    BOOST_LOG_TRIVIAL(info) << "Generation complete, goodbye!";

    return 0;
}
