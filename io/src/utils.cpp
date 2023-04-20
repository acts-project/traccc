/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/utils.hpp"

// System include(s).
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace traccc::io {

const std::string& data_directory() {

    // Establish the directory name only once for the application.
    static const std::string data_dir = [] {
        // Look for an environment variable specifying the directory name.
        const char* env_dir = std::getenv("TRACCC_TEST_DATA_DIR");
        if (env_dir == nullptr) {
            // If an environment variable is not set, rely on the definition
            // coming from the build system.
            return std::string(TRACCC_TEST_DATA_DIR) + "/";
        }
        // Initialise the directory name using the environment variable.
        return std::string(env_dir) + "/";
    }();

    // Return the previously initialised variable.
    return data_dir;
}

std::string get_event_filename(std::size_t event, std::string_view suffix) {

    std::ostringstream stream;
    stream << "event" << std::setfill('0') << std::setw(9) << event << suffix;
    return stream.str();
}

}  // namespace traccc::io
