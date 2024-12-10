/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <fstream>
#include <iostream>
#include <sstream>

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

using field_t = covfie::field<covfie::backend::affine<
    covfie::backend::nearest_neighbour<covfie::backend::strided<
        covfie::vector::size3,
        covfie::backend::array<covfie::vector::float3>>>>>;

field_t read_bfield(const std::string & fn)
{
    std::ifstream f;

    float minx = std::numeric_limits<float>::max();
    float maxx = std::numeric_limits<float>::lowest();
    float miny = std::numeric_limits<float>::max();
    float maxy = std::numeric_limits<float>::lowest();
    float minz = std::numeric_limits<float>::max();
    float maxz = std::numeric_limits<float>::lowest();

    {
        BOOST_LOG_TRIVIAL(info)
            << "Opening magnetic field to compute field limits";

        f.open(fn);

        if (!f.good()) {
            BOOST_LOG_TRIVIAL(fatal)
                << "Failed to open input file " << fn << "!";
            std::exit(1);
        }

        std::string line;

        BOOST_LOG_TRIVIAL(info) << "Skipping the first line (header)";

        std::getline(f, line);

        float xp, yp, zp;
        float Bx, By, Bz;

        (void)Bx, (void)By, (void)Bz;

        std::size_t n_lines = 0;

        BOOST_LOG_TRIVIAL(info)
            << "Iterating over lines in the magnetic field file";

        /*
         * Read every line, and update our current minima and maxima
         * appropriately.
         */
        while (std::getline(f, line)) {
            std::string word;
            std::stringstream ss(line);
            std::getline(ss, word, ',');
            xp = static_cast<float>(std::atof(word.c_str()));
            std::getline(ss, word, ',');
            yp = static_cast<float>(std::atof(word.c_str()));
            std::getline(ss, word, ',');
            zp = static_cast<float>(std::atof(word.c_str()));
            std::getline(ss, word, ',');
            Bx = static_cast<float>(std::atof(word.c_str()));
            std::getline(ss, word, ',');
            By = static_cast<float>(std::atof(word.c_str()));
            std::getline(ss, word, ',');
            Bz = static_cast<float>(std::atof(word.c_str()));

            minx = std::min(minx, xp);
            maxx = std::max(maxx, xp);

            miny = std::min(miny, yp);
            maxy = std::max(maxy, yp);

            minz = std::min(minz, zp);
            maxz = std::max(maxz, zp);

            ++n_lines;
        }

        BOOST_LOG_TRIVIAL(info)
            << "Read " << n_lines << " lines of magnetic field data";

        BOOST_LOG_TRIVIAL(info) << "Closing magnetic field file";

        f.close();
    }

    BOOST_LOG_TRIVIAL(info)
        << "Field dimensions in x = [" << minx << ", " << maxx << "]";
    BOOST_LOG_TRIVIAL(info)
        << "Field dimensions in y = [" << miny << ", " << maxy << "]";
    BOOST_LOG_TRIVIAL(info)
        << "Field dimensions in z = [" << minz << ", " << maxz << "]";

    BOOST_LOG_TRIVIAL(info)
        << "Assuming sample spacing of 100.0 in each dimension";

    /*
     * Now that we have the limits of our field, compute the size in each
     * dimension.
     */
    std::size_t sx =
        static_cast<std::size_t>(std::lround((maxx - minx) / 100.0)) + 1;
    std::size_t sy =
        static_cast<std::size_t>(std::lround((maxy - miny) / 100.0)) + 1;
    std::size_t sz =
        static_cast<std::size_t>(std::lround((maxz - minz) / 100.0)) + 1;

    BOOST_LOG_TRIVIAL(info)
        << "Magnetic field size is " << sx << "x" << sy << "x" << sz;

    BOOST_LOG_TRIVIAL(info) << "Constructing matching vector field...";

    covfie::algebra::affine<3> translation =
        covfie::algebra::affine<3>::translation(-minx, -miny, -minz);
    covfie::algebra::affine<3> scaling = covfie::algebra::affine<3>::scaling(
        static_cast<float>(sx - 1) / (maxx - minx),
        static_cast<float>(sy - 1) / (maxy - miny),
        static_cast<float>(sz - 1) / (maxz - minz)
    );

    field_t field(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t(scaling * translation),
        field_t::backend_t::backend_t::configuration_t{},
        field_t::backend_t::backend_t::backend_t::configuration_t{sx, sy, sz}
    ));
    field_t::view_t fv(field);

    {
        BOOST_LOG_TRIVIAL(info) << "Re-opening magnetic field to gather data";

        f.open(fn);

        if (!f.good()) {
            BOOST_LOG_TRIVIAL(fatal)
                << "Failed to open input file " << fn << "!";
            std::exit(1);
        }

        std::string line;

        BOOST_LOG_TRIVIAL(info) << "Skipping the first line (header)";

        std::getline(f, line);

        float xp, yp, zp;
        float Bx, By, Bz;

        std::size_t n_lines = 0;

        BOOST_LOG_TRIVIAL(info)
            << "Iterating over lines in the magnetic field file";

        /*
         * Read every line, and update our current minima and maxima
         * appropriately.
         */
        while (std::getline(f, line)) {
            std::string word;
            std::stringstream ss(line);
            std::getline(ss, word, ',');
            xp = static_cast<float>(std::atof(word.c_str()));
            std::getline(ss, word, ',');
            yp = static_cast<float>(std::atof(word.c_str()));
            std::getline(ss, word, ',');
            zp = static_cast<float>(std::atof(word.c_str()));
            std::getline(ss, word, ',');
            Bx = static_cast<float>(std::atof(word.c_str()));
            std::getline(ss, word, ',');
            By = static_cast<float>(std::atof(word.c_str()));
            std::getline(ss, word, ',');
            Bz = static_cast<float>(std::atof(word.c_str()));

            field_t::view_t::output_t & p = fv.at(xp, yp, zp);

            p[0] = Bx * 0.000299792458f;
            p[1] = By * 0.000299792458f;
            p[2] = Bz * 0.000299792458f;

            n_lines++;
        }

        BOOST_LOG_TRIVIAL(info)
            << "Read " << n_lines << " lines of magnetic field data";

        BOOST_LOG_TRIVIAL(info) << "Closing magnetic field file";

        f.close();
    }

    return field;
}

int main(int argc, char ** argv)
{
    boost::program_options::variables_map vm;
    parse_opts(argc, argv, vm);

    BOOST_LOG_TRIVIAL(info) << "Welcome to the covfie magnetic field renderer!";
    BOOST_LOG_TRIVIAL(info) << "Using magnetic field file \""
                            << vm["input"].as<std::string>() << "\"";
    BOOST_LOG_TRIVIAL(info) << "Starting read of input file...";

    field_t fb = read_bfield(vm["input"].as<std::string>());

    BOOST_LOG_TRIVIAL(info) << "Writing magnetic field to file \""
                            << vm["output"].as<std::string>() << "\"...";

    std::ofstream fs(vm["output"].as<std::string>(), std::ofstream::binary);

    if (!fs.good()) {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to open output file "
                                 << vm["output"].as<std::string>() << "!";
        std::exit(1);
    }

    fb.dump(fs);

    fs.close();

    BOOST_LOG_TRIVIAL(info) << "Rendering complete, goodbye!";

    return 0;
}
