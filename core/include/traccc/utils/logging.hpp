/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <Acts/Utilities/Logger.hpp>

namespace traccc {
namespace Logging = ::Acts::Logging;

using Logger = ::Acts::Logger;

inline std::unique_ptr<const Logger> getDefaultLogger(
    const std::string& name, const Logging::Level& lvl,
    std::ostream* log_stream = &std::cout) {
    return ::Acts::getDefaultLogger(name, lvl, log_stream);
}

inline const Logger& getDummyLogger() {
    return ::Acts::getDummyLogger();
}
}  // namespace traccc

#define TRACCC_LOCAL_LOGGER(x) ACTS_LOCAL_LOGGER(x)
#define TRACCC_LOG(level, x) ACTS_LOG(level, log)
#define TRACCC_VERBOSE(x) ACTS_VERBOSE(x)
#define TRACCC_DEBUG(x) ACTS_DEBUG(x)
#define TRACCC_INFO(x) ACTS_INFO(x)
#define TRACCC_WARNING(x) ACTS_WARNING(x)
#define TRACCC_ERROR(x) ACTS_ERROR(x)
#define TRACCC_FATAL(x) ACTS_FATAL(x)
