/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>

/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <cstddef>
#include <fstream>
#include <iostream>

#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>

#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/core/utility/nd_size.hpp>

void parse_opts(
    int argc, char * argv[], boost::program_options::variables_map & vm
)
{
    boost::program_options::options_description opts("general options");

    opts.add_options()("help", "produce help message")(
        "input,i",
        boost::program_options::value<std::string>()->required(),
        "input vector field to read"
    )("output,o",
      boost::program_options::value<std::string>()->required(),
      "output vector field to write"
    )("axis,a",
      boost::program_options::value<std::string>()->required(),
      "axis along which to slice (\"x\", \"y\", or \"z\")"
    )("slice,s",
      boost::program_options::value<unsigned long>()->required(),
      "slice coordinate along the specified axis");

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
        BOOST_LOG_TRIVIAL(fatal) << e.what();
        std::exit(1);
    }

    if (vm["axis"].as<std::string>() != "x" &&
        vm["axis"].as<std::string>() != "y" &&
        vm["axis"].as<std::string>() != "z")
    {
        BOOST_LOG_TRIVIAL(fatal) << "Axis specification must be x, y, or z!";
        std::exit(1);
    }
}

int main(int argc, char ** argv)
{
    using core_t = covfie::backend::strided<
        covfie::vector::size3,
        covfie::backend::array<covfie::vector::float3>>;
    using field_t1 =
        covfie::field<covfie::backend::affine<covfie::backend::linear<core_t>>>;
    using field_t2 = covfie::field<covfie::backend::strided<
        covfie::vector::size2,
        covfie::backend::array<covfie::vector::float3>>>;

    boost::program_options::variables_map vm;
    parse_opts(argc, argv, vm);

    BOOST_LOG_TRIVIAL(info) << "Welcome to the covfie vector field slicer!";
    BOOST_LOG_TRIVIAL(info) << "Using vector field file \""
                            << vm["input"].as<std::string>() << "\"";
    BOOST_LOG_TRIVIAL(info) << "Starting read of input file...";

    std::ifstream ifs(vm["input"].as<std::string>(), std::ifstream::binary);

    if (!ifs.good()) {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to open input file "
                                 << vm["input"].as<std::string>() << "!";
        std::exit(1);
    }

    field_t1 f(ifs);
    ifs.close();

    covfie::field<core_t> cf(
        covfie::make_parameter_pack(f.backend().get_backend().get_backend())
    );
    covfie::field<core_t>::view_t cfv(cf);

    BOOST_LOG_TRIVIAL(info) << "Building new output vector field...";

    covfie::utility::nd_size<3> in_size =
        f.backend().get_backend().get_backend().get_configuration();
    covfie::utility::nd_size<2> out_size{0u, 0u};

    if (vm["axis"].as<std::string>() == "x") {
        out_size = {in_size[1], in_size[2]};
    } else if (vm["axis"].as<std::string>() == "y") {
        out_size = {in_size[0], in_size[2]};
    } else if (vm["axis"].as<std::string>() == "z") {
        out_size = {in_size[0], in_size[1]};
    }

    field_t2 of(covfie::make_parameter_pack(
        field_t2::backend_t::configuration_t{out_size}
    ));

    BOOST_LOG_TRIVIAL(info) << "Creating vector field views...";

    field_t2::view_t ofv(of);

    BOOST_LOG_TRIVIAL(info) << "Slicing vector field...";

    using scalar_t =
        decltype(ofv)::field_t::backend_t::contravariant_input_t::scalar_t;

    if (vm["axis"].as<std::string>() == "x") {
        for (scalar_t x = 0; x < out_size[0]; ++x) {
            for (scalar_t y = 0; y < out_size[1]; ++y) {
                ofv.at(x, y) = cfv.at(vm["slice"].as<scalar_t>(), x, y);
            }
        }
    } else if (vm["axis"].as<std::string>() == "y") {
        for (scalar_t x = 0; x < out_size[0]; ++x) {
            for (scalar_t y = 0; y < out_size[1]; ++y) {
                ofv.at(x, y) = cfv.at(x, vm["slice"].as<scalar_t>(), y);
            }
        }
    } else if (vm["axis"].as<std::string>() == "z") {
        for (scalar_t x = 0; x < out_size[0]; ++x) {
            for (scalar_t y = 0; y < out_size[1]; ++y) {
                ofv.at(x, y) = cfv.at(x, y, vm["slice"].as<scalar_t>());
            }
        }
    }

    BOOST_LOG_TRIVIAL(info) << "Saving result to file \""
                            << vm["output"].as<std::string>() << "\"...";

    std::ofstream ofs(vm["output"].as<std::string>(), std::ifstream::binary);

    if (!ofs.good()) {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to open output file "
                                 << vm["output"].as<std::string>() << "!";
        std::exit(1);
    }

    of.dump(ofs);
    ofs.close();

    BOOST_LOG_TRIVIAL(info) << "Procedure complete, goodbye!";
}
