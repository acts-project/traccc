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
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/nearest_neighbour.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>

using basic_backend_t = covfie::backend::strided<
    covfie::vector::size3,
    covfie::backend::array<covfie::vector::float3>>;

using field_nn_backend_t = covfie::backend::affine<
    covfie::backend::nearest_neighbour<basic_backend_t>>;

using field_li_backend_t =
    covfie::backend::affine<covfie::backend::linear<basic_backend_t>>;

TEST(TestAtlasLikeIO, WriteReadAtlasLike)
{
    covfie::field<basic_backend_t> bf(
        covfie::make_parameter_pack_for<covfie::field<basic_backend_t>>(
            {2u, 2u, 2u}, {8u}
        )
    );
    covfie::field<basic_backend_t>::view_t bv(bf);

    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 2; y++) {
            for (int z = 0; z < 2; z++) {
                bv.at(x, y, z) = {
                    static_cast<float>(x),
                    static_cast<float>(y),
                    static_cast<float>(z)};
            }
        }
    }

    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 2; y++) {
            for (int z = 0; z < 2; z++) {
                covfie::field<field_nn_backend_t>::output_t & p =
                    bv.at(x, y, z);
                EXPECT_EQ(p[0], static_cast<float>(x));
                EXPECT_EQ(p[1], static_cast<float>(y));
                EXPECT_EQ(p[2], static_cast<float>(z));
            }
        }
    }

    covfie::algebra::affine<3> translation =
        covfie::algebra::affine<3>::translation(0.0f, 0.0f, 0.0f);

    covfie::field<field_nn_backend_t> nnf(covfie::make_parameter_pack(
        field_nn_backend_t::configuration_t(translation),
        field_nn_backend_t::backend_t::configuration_t{},
        std::move(bf.backend())
    ));

    boost::filesystem::path ofile = get_tmp_file();

    std::ofstream ofs(ofile.native(), std::ofstream::binary);

    if (!ofs.good()) {
        throw std::runtime_error("Invalid file was somehow opened!");
    }

    nnf.dump(ofs);
    ofs.close();

    std::ifstream ifs(ofile.native(), std::ifstream::binary);

    if (!ifs.good()) {
        throw std::runtime_error("Invalid file was somehow opened!");
    }

    covfie::field<field_nn_backend_t> nnnf(ifs);
    ifs.close();

    covfie::field<field_nn_backend_t>::view_t nnnv(nnnf);

    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 2; y++) {
            for (int z = 0; z < 2; z++) {
                covfie::field<field_nn_backend_t>::output_t & p =
                    nnnv.at(x, y, z);
                EXPECT_EQ(p[0], static_cast<float>(x));
                EXPECT_EQ(p[1], static_cast<float>(y));
                EXPECT_EQ(p[2], static_cast<float>(z));
            }
        }
    }
}

TEST(TestAtlasLikeIO, WriteReadAtlasLikeChangeInterpolation)
{
    covfie::field<basic_backend_t> bf(
        covfie::make_parameter_pack_for<covfie::field<basic_backend_t>>(
            {3u, 3u, 3u}, {27u}
        )
    );
    covfie::field<basic_backend_t>::view_t bv(bf);

    for (int x = 0; x < 3; x++) {
        for (int y = 0; y < 3; y++) {
            for (int z = 0; z < 3; z++) {
                bv.at(x, y, z) = {
                    static_cast<float>(x),
                    static_cast<float>(y),
                    static_cast<float>(z)};
            }
        }
    }

    for (int x = 0; x < 3; x++) {
        for (int y = 0; y < 3; y++) {
            for (int z = 0; z < 3; z++) {
                covfie::field<field_nn_backend_t>::output_t p = bv.at(x, y, z);
                EXPECT_EQ(p[0], static_cast<float>(x));
                EXPECT_EQ(p[1], static_cast<float>(y));
                EXPECT_EQ(p[2], static_cast<float>(z));
            }
        }
    }

    covfie::algebra::affine<3> translation =
        covfie::algebra::affine<3>::translation(0.0f, 0.0f, 0.0f);

    covfie::field<field_nn_backend_t> nnf(covfie::make_parameter_pack(
        field_nn_backend_t::configuration_t(translation),
        field_nn_backend_t::backend_t::configuration_t{},
        std::move(bf.backend())
    ));

    boost::filesystem::path ofile = get_tmp_file();

    std::ofstream ofs(ofile.native(), std::ofstream::binary);

    if (!ofs.good()) {
        throw std::runtime_error("Invalid file was somehow opened!");
    }

    nnf.dump(ofs);
    ofs.close();

    std::ifstream ifs(ofile.native(), std::ifstream::binary);

    if (!ifs.good()) {
        throw std::runtime_error("Invalid file was somehow opened!");
    }

    covfie::field<field_li_backend_t> nnnf(ifs);
    ifs.close();

    covfie::field<field_li_backend_t>::view_t nnnv(nnnf);

    for (float x = 0.f; x < 2.f; x += 0.321f) {
        for (float y = 0.f; y < 2.f; y += 0.321f) {
            for (float z = 0.f; z < 2.f; z += 0.321f) {
                covfie::field<field_li_backend_t>::output_t p =
                    nnnv.at(x, y, z);
                EXPECT_NEAR(p[0], x, 0.01f);
                EXPECT_NEAR(p[1], y, 0.01f);
                EXPECT_NEAR(p[2], z, 0.01f);
            }
        }
    }
}
