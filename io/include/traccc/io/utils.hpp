/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <cstddef>
#include <string>
#include <string_view>

namespace traccc::io {

/// Get the data directory to use in the application
///
/// @return The directory name to find data files in
///
const std::string& data_directory();

/// Get the name of a data file for a specific event ID
///
/// @param event The event number to get the file name for
/// @param suffix A suffix for the file name
/// @return A standardized name for a data file to use as input.
///
std::string get_event_filename(std::size_t event, std::string_view suffix);

}  // namespace traccc::io