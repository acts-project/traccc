#pragma once

#include <iomanip>
#include <sstream>

namespace traccc {

const std::string &data_directory() {
    static const std::string data_dir = [] {
        auto env_d_d = std::getenv("TRACCC_TEST_DATA_DIR");
        if (env_d_d == nullptr) {
            throw std::ios_base::failure(
                "Test data directory not found. Please set "
                "TRACCC_TEST_DATA_DIR.");
        }
        std::string data_dir = std::string(env_d_d);
        return data_dir.append("/");
    }();

    return data_dir;
}

std::string get_event_filename(size_t event, const std::string &suffix) {
    std::stringstream stream;
    stream << "event";
    stream << std::setfill('0') << std::setw(9) << event;
    stream << suffix;
    return stream.str();
}

}  // namespace traccc