/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Acts include(s).
#include <Acts/Utilities/Logger.hpp>

namespace traccc {

/// Use the @c Acts::Logging namespace.
namespace Logging = ::Acts::Logging;

/// Use the @c Acts::Logger type.
using Logger = ::Acts::Logger;

/// Construct a logger with default settings for this project
///
/// @param name the name of the log writer
/// @param lvl the log level
/// @param log_stream the stream to write the log to
///
/// @return a unique pointer to the logger
///
std::unique_ptr<const Logger> getDefaultLogger(
    const std::string& name, const Logging::Level& lvl = Logging::INFO,
    std::ostream* log_stream = &std::cout);

/// Construct a dummy logger that does nothing
///
/// @return a reference to the dummy logger
///
const Logger& getDummyLogger();

}  // namespace traccc

#define TRACCC_LOCAL_LOGGER(x) ACTS_LOCAL_LOGGER(x)
#define TRACCC_LOG(level, x) ACTS_LOG(level, log)
#define TRACCC_VERBOSE(x) ACTS_VERBOSE(x)
#define TRACCC_DEBUG(x) ACTS_DEBUG(x)
#define TRACCC_INFO(x) ACTS_INFO(x)
#define TRACCC_WARNING(x) ACTS_WARNING(x)
#define TRACCC_ERROR(x) ACTS_ERROR(x)
#define TRACCC_FATAL(x) ACTS_FATAL(x)
