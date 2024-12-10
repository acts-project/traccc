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

TEST(TestArrayBinaryIO, WriteReadFloatFloat)
{
    using field_t1 =
        covfie::field<covfie::backend::array<covfie::vector::float1>>;
    using field_t2 = field_t1;

    field_t1 f(covfie::make_parameter_pack(field_t1::backend_t::configuration_t{
        5ul}));

    field_t1::view_t fv(f);

    for (std::size_t i = 0ul; i < 5ul; ++i) {
        field_t1::output_t & p = fv.at(i);
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

    field_t2 nf(ifs);
    ifs.close();

    field_t2::view_t nfv(nf);

    for (std::size_t i = 0ul; i < 5ul; ++i) {
        field_t2::output_t & p = nfv.at(i);
        EXPECT_EQ(p[0], static_cast<float>(i));
    }
}

TEST(TestArrayBinaryIO, WriteReadDoubleDouble)
{
    using field_t1 =
        covfie::field<covfie::backend::array<covfie::vector::double1>>;
    using field_t2 = field_t1;

    field_t1 f(covfie::make_parameter_pack(field_t1::backend_t::configuration_t{
        5ul}));

    field_t1::view_t fv(f);

    for (std::size_t i = 0ul; i < 5ul; ++i) {
        field_t1::output_t & p = fv.at(i);
        p[0] = static_cast<double>(i);
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

    field_t2 nf(ifs);
    ifs.close();

    field_t2::view_t nfv(nf);

    for (std::size_t i = 0ul; i < 5ul; ++i) {
        field_t2::output_t & p = nfv.at(i);
        EXPECT_EQ(p[0], static_cast<double>(i));
    }
}

TEST(TestArrayBinaryIO, WriteReadFloatDouble)
{
    using field_t1 =
        covfie::field<covfie::backend::array<covfie::vector::float1>>;
    using field_t2 =
        covfie::field<covfie::backend::array<covfie::vector::double1>>;
    ;

    field_t1 f(covfie::make_parameter_pack(field_t1::backend_t::configuration_t{
        5ul}));

    field_t1::view_t fv(f);

    for (std::size_t i = 0ul; i < 5ul; ++i) {
        field_t1::output_t & p = fv.at(i);
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

    field_t2 nf(ifs);
    ifs.close();

    field_t2::view_t nfv(nf);

    for (std::size_t i = 0ul; i < 5ul; ++i) {
        field_t2::output_t & p = nfv.at(i);
        EXPECT_EQ(p[0], static_cast<double>(i));
    }
}

TEST(TestArrayBinaryIO, WriteReadDoubleFloat)
{
    using field_t1 =
        covfie::field<covfie::backend::array<covfie::vector::double1>>;
    using field_t2 =
        covfie::field<covfie::backend::array<covfie::vector::float1>>;
    ;

    field_t1 f(covfie::make_parameter_pack(field_t1::backend_t::configuration_t{
        5ul}));

    field_t1::view_t fv(f);

    for (std::size_t i = 0ul; i < 5ul; ++i) {
        field_t1::output_t & p = fv.at(i);
        p[0] = static_cast<double>(i);
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

    field_t2 nf(ifs);
    ifs.close();

    field_t2::view_t nfv(nf);

    for (std::size_t i = 0ul; i < 5ul; ++i) {
        field_t2::output_t & p = nfv.at(i);
        EXPECT_EQ(p[0], static_cast<float>(i));
    }
}
