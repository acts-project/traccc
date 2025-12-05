/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "device_backend.hpp"

// System include(s).
#include <string_view>

namespace traccc {

/// Helper function implementing a device reconstruction validation application
///
/// Applications that used to be called "traccc_seq_example_<platform>" in the
/// past.
///
/// @tparam backend_t The device backend type to use
///
/// @param logger_name The name to use for the logger
/// @param description A description for the application
/// @param argc The @c argc argument coming from @c main(...)
/// @param argv The @c argc argument coming from @c main(...)
///
/// @return The value to be returned from @c main(...)
///
template <concepts::device_backend backend_t>
int device_reconstruction_validation(std::string_view logger_name,
                                     std::string_view description, int argc,
                                     char* argv[]);

}  // namespace traccc

// Include the implementation.
#include "device_reconstruction_validation.ipp"
