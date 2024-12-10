/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <cstdlib>
#include <fstream>

#include <covfie/benchmark/test_field.hpp>

std::unique_ptr<data_field_t> TEST_FIELD;

const data_field_t & get_test_field()
{
    if (!TEST_FIELD) {
        char * file_name = std::getenv("COVFIE_BENCHMARK_FIELD");

        if (file_name) {
            std::ifstream ifs(file_name, std::ifstream::binary);
            if (!ifs.good()) {
                throw std::runtime_error(
                    "Failed to open testing vector field file!"
                );
            }
            TEST_FIELD = std::make_unique<data_field_t>(ifs);
        } else {
            throw std::runtime_error(
                "Environment variable \"COVFIE_BENCHMARK_FIELD\" is not "
                "set!"
            );
        }
    }

    return *TEST_FIELD.get();
}
