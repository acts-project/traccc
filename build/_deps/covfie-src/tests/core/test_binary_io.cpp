/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <fstream>

#include <boost/filesystem.hpp>
#include <gtest/gtest.h>
#include <tmp_file.hpp>

#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/field.hpp>

TEST(TestBinaryIO, WriteRead1DSingleFloatBuilder)
{
    using field_t =
        covfie::field<covfie::backend::array<covfie::vector::float1>>;

    field_t f(covfie::make_parameter_pack(field_t::backend_t::configuration_t{
        5ul}));

    field_t::view_t fv(f);

    for (std::size_t i = 0ul; i < 5ul; ++i) {
        field_t::output_t & p = fv.at(i);
        p[0] = static_cast<float>(i);
    }

    boost::filesystem::path ofile = get_tmp_file();

    std::ofstream ofs(ofile.native(), std::ofstream::binary);

    if (!ofs.good()) {
        throw std::runtime_error("Invalid file was somehow opened!");
    }

    f.dump(ofs);
    ofs.close();

    std::ifstream ifs(ofile.native(), std::ifstream::binary);

    if (!ifs.good()) {
        throw std::runtime_error("Invalid file was somehow opened!");
    }

    field_t nf(ifs);
    ifs.close();

    field_t::view_t nfv(nf);

    for (std::size_t i = 0ul; i < 5ul; ++i) {
        field_t::output_t & p = nfv.at(i);
        EXPECT_EQ(p[0], static_cast<float>(i));
    }
}

// TEST(TestBinaryIO, WriteRead1DArrayFloat)
// {
//     covfie::field_builder<float, 1, float, 3> fb(5u);

//     for (std::size_t i = 0; i < 5u; ++i) {
//         decltype(fb)::output_type & p = fb.at_integral(i);
//         p[0] = static_cast<float>(3 * i);
//         p[1] = static_cast<float>(3 * i + 1);
//         p[2] = static_cast<float>(3 * i + 2);
//     }

//     boost::filesystem::path ofile =
//         boost::filesystem::temp_directory_path() /
//         boost::filesystem::unique_path("covfie_test_%%%%_%%%%_%%%%_%%%%.covfie"
//         );

//     covfie::field<float, 1, float, 3, covfie::backend::file_writer>(
//         fb, ofile.native()
//     );

//     covfie::field_builder<float, 1, float, 3> nfb =
//         covfie::io::read_binary_file<float, 1, float, 3>(ofile.native());

//     EXPECT_EQ(nfb.at_integral(0u)[0], 0.0f);
//     EXPECT_EQ(nfb.at_integral(0u)[1], 1.0f);
//     EXPECT_EQ(nfb.at_integral(0u)[2], 2.0f);

//     EXPECT_EQ(nfb.at_integral(1u)[0], 3.0f);
//     EXPECT_EQ(nfb.at_integral(1u)[1], 4.0f);
//     EXPECT_EQ(nfb.at_integral(1u)[2], 5.0f);

//     EXPECT_EQ(nfb.at_integral(2u)[0], 6.0f);
//     EXPECT_EQ(nfb.at_integral(2u)[1], 7.0f);
//     EXPECT_EQ(nfb.at_integral(2u)[2], 8.0f);

//     EXPECT_EQ(nfb.at_integral(3u)[0], 9.0f);
//     EXPECT_EQ(nfb.at_integral(3u)[1], 10.0f);
//     EXPECT_EQ(nfb.at_integral(3u)[2], 11.0f);

//     EXPECT_EQ(nfb.at_integral(4u)[0], 12.0f);
//     EXPECT_EQ(nfb.at_integral(4u)[1], 13.0f);
//     EXPECT_EQ(nfb.at_integral(4u)[2], 14.0f);
// }

// TEST(TestBinaryIO, WriteRead3DSingleFloat)
// {
//     covfie::field_builder<float, 3, float, 1> fb(3u, 4u, 5u);

//     for (std::size_t x = 0; x < 3u; ++x) {
//         for (std::size_t y = 0; y < 4u; ++y) {
//             for (std::size_t z = 0; z < 5u; ++z) {
//                 decltype(fb)::output_type & p = fb.at_integral(x, y, z);
//                 p[0] = static_cast<float>(100 * x + 10 * y + z);
//             }
//         }
//     }

//     boost::filesystem::path ofile =
//         boost::filesystem::temp_directory_path() /
//         boost::filesystem::unique_path("covfie_test_%%%%_%%%%_%%%%_%%%%.covfie"
//         );

//     covfie::field<float, 3, float, 1, covfie::backend::file_writer>(
//         fb, ofile.native()
//     );

//     covfie::field_builder<float, 3, float, 1> nfb =
//         covfie::io::read_binary_file<float, 3, float, 1>(ofile.native());

//     EXPECT_EQ(nfb.at_integral(0u, 0u, 0u)[0], 0.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 0u, 1u)[0], 1.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 0u, 2u)[0], 2.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 0u, 3u)[0], 3.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 0u, 4u)[0], 4.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 1u, 0u)[0], 10.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 1u, 1u)[0], 11.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 1u, 2u)[0], 12.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 1u, 3u)[0], 13.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 1u, 4u)[0], 14.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 2u, 0u)[0], 20.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 2u, 1u)[0], 21.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 2u, 2u)[0], 22.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 2u, 3u)[0], 23.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 2u, 4u)[0], 24.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 3u, 0u)[0], 30.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 3u, 1u)[0], 31.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 3u, 2u)[0], 32.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 3u, 3u)[0], 33.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 3u, 4u)[0], 34.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 0u, 0u)[0], 100.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 0u, 1u)[0], 101.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 0u, 2u)[0], 102.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 0u, 3u)[0], 103.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 0u, 4u)[0], 104.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 1u, 0u)[0], 110.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 1u, 1u)[0], 111.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 1u, 2u)[0], 112.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 1u, 3u)[0], 113.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 1u, 4u)[0], 114.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 2u, 0u)[0], 120.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 2u, 1u)[0], 121.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 2u, 2u)[0], 122.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 2u, 3u)[0], 123.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 2u, 4u)[0], 124.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 3u, 0u)[0], 130.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 3u, 1u)[0], 131.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 3u, 2u)[0], 132.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 3u, 3u)[0], 133.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 3u, 4u)[0], 134.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 0u, 0u)[0], 200.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 0u, 1u)[0], 201.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 0u, 2u)[0], 202.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 0u, 3u)[0], 203.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 0u, 4u)[0], 204.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 1u, 0u)[0], 210.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 1u, 1u)[0], 211.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 1u, 2u)[0], 212.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 1u, 3u)[0], 213.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 1u, 4u)[0], 214.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 2u, 0u)[0], 220.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 2u, 1u)[0], 221.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 2u, 2u)[0], 222.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 2u, 3u)[0], 223.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 2u, 4u)[0], 224.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 3u, 0u)[0], 230.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 3u, 1u)[0], 231.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 3u, 2u)[0], 232.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 3u, 3u)[0], 233.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 3u, 4u)[0], 234.0f);
// }
