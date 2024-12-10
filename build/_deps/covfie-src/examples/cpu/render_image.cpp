/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
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
#include <covfie/core/utility/nd_size.hpp>

#include "bitmap.hpp"

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
      "output bitmap image to write");

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
}

int main(int argc, char ** argv)
{
    using field_t = covfie::field<covfie::backend::strided<
        covfie::vector::size2,
        covfie::backend::array<covfie::vector::float3>>>;

    boost::program_options::variables_map vm;
    parse_opts(argc, argv, vm);

    BOOST_LOG_TRIVIAL(info) << "Welcome to the covfie image renderer!";
    BOOST_LOG_TRIVIAL(info) << "Using vector field file \""
                            << vm["input"].as<std::string>() << "\"";
    BOOST_LOG_TRIVIAL(info) << "Starting read of input file...";

    std::ifstream ifs(vm["input"].as<std::string>(), std::ifstream::binary);

    if (!ifs.good()) {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to open input file "
                                 << vm["input"].as<std::string>() << "!";
        std::exit(1);
    }

    field_t f(ifs);
    ifs.close();

    BOOST_LOG_TRIVIAL(info) << "Creating magnetic field view...";

    field_t::view_t fv(f);

    BOOST_LOG_TRIVIAL(info) << "Allocating memory for output image...";

    covfie::utility::nd_size<2> im_size = f.backend().get_configuration();

    std::unique_ptr<unsigned char[]> img =
        std::make_unique<unsigned char[]>(im_size[1] * im_size[0]);

    BOOST_LOG_TRIVIAL(info) << "Rendering vector field to image...";

    for (std::size_t x = 0; x < im_size[1]; ++x) {
        for (std::size_t y = 0; y < im_size[0]; ++y) {
            field_t::view_t::output_t p = fv.at(x, y);

            img[im_size[1] * y + x] = static_cast<unsigned char>(std::lround(
                255.f *
                std::min(
                    std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]), 1.0f
                )
            ));
        }
    }

    BOOST_LOG_TRIVIAL(info) << "Saving image to file \""
                            << vm["output"].as<std::string>() << "\"...";

    render_bitmap(
        img.get(),
        static_cast<unsigned int>(im_size[1]),
        static_cast<unsigned int>(im_size[0]),
        vm["output"].as<std::string>()
    );

    BOOST_LOG_TRIVIAL(info) << "Procedure complete, goodbye!";
}
