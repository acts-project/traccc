/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace traccc {

inline const std::string &data_directory() {
    static const std::string data_dir = [] {
        auto env_d_d = std::getenv("TRACCC_TEST_DATA_DIR");
        if (env_d_d == nullptr) {
            throw std::ios_base::failure(
                "Test data directory not found. Please set "
                "TRACCC_TEST_DATA_DIR.");
        }
        return std::string(env_d_d) + "/";
    }();

    return data_dir;
}

inline std::string get_event_filename(size_t event, const std::string &suffix) {
    std::stringstream stream;
    stream << "event";
    stream << std::setfill('0') << std::setw(9) << event;
    stream << suffix;
    return stream.str();
}

}  // namespace traccc