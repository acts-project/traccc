/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <boost/filesystem.hpp>
#include <tmp_file.hpp>

boost::filesystem::path get_tmp_file()
{
    return boost::filesystem::temp_directory_path() /
           boost::filesystem::unique_path(
               "covfie_test_%%%%_%%%%_%%%%_%%%%.covfie"
           );
}
