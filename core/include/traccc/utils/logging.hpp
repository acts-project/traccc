/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Acts include(s).
#include <Acts/Utilities/Logger.hpp>

// detray include(s)
#include <detray/utils/log.hpp>

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

/// @TODO: Temporary usage of detray logger, until the ACTS logger can be
/// used/passed in host-device code

// Exclude SYCL and HIP builds, which run into linker errors
#if defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION) || \
    defined(__HIP__)
#define __EXCLUDE_LOGS__
#endif

#if !defined(__EXCLUDE_LOGS__)
// Host log messages in host-device code (@TODO: This will currently claim to
// come from "DETRAY")
#define TRACCC_VERBOSE_HOST(x) DETRAY_VERBOSE_HOST(x)
#define TRACCC_DEBUG_HOST(x) DETRAY_DEBUG_HOST(x)
#define TRACCC_INFO_HOST(x) DETRAY_INFO_HOST(x)
#define TRACCC_WARNING_HOST(x) DETRAY_WARN_HOST(x)
#define TRACCC_ERROR_HOST(x) DETRAY_ERROR_HOST(x)
#define TRACCC_FATAL_HOST(x) DETRAY_FATAL_HOST(x)

// Host-device log messages for host-device code
#define TRACCC_WARNING_HOST_DEVICE(x, ...) \
    DETRAY_LOG_PRINTF("TRACCC", "WARNING", x, __VA_ARGS__)
#define TRACCC_ERROR_HOST_DEVICE(x, ...) \
    DETRAY_LOG_PRINTF("TRACCC", "ERROR", x, __VA_ARGS__)
#define TRACCC_FATAL_HOST_DEVICE(x, ...) \
    DETRAY_LOG_PRINTF("TRACCC", "FATAL", x, __VA_ARGS__)
#else
#define TRACCC_VERBOSE_HOST(x)
#define TRACCC_DEBUG_HOST(x)
#define TRACCC_INFO_HOST(x)
#define TRACCC_WARNING_HOST(x)
#define TRACCC_ERROR_HOST(x)
#define TRACCC_FATAL_HOST(x)

#define TRACCC_WARNING_HOST_DEVICE(x, ...)
#define TRACCC_ERROR_HOST_DEVICE(x, ...)
#define TRACCC_FATAL_HOST_DEVICE(x, ...)
#endif  // defined(__EXCLUDE_LOGS__)

#if DETRAY_LOG_LVL > 0 && !defined(__EXCLUDE_LOGS__)
#define TRACCC_INFO_HOST_DEVICE(x, ...) \
    DETRAY_LOG_PRINTF("TRACCC", "INFO", x, __VA_ARGS__)
#else
#define TRACCC_INFO_HOST_DEVICE(x, ...)
#endif

#if DETRAY_LOG_LVL > 1 && !defined(__EXCLUDE_LOGS__)
#define TRACCC_VERBOSE_HOST_DEVICE(x, ...) \
    DETRAY_LOG_PRINTF("TRACCC", "VERBOSE", x, __VA_ARGS__)
#else
#define TRACCC_VERBOSE_HOST_DEVICE(x, ...)
#endif

#if DETRAY_LOG_LVL > 2 && !defined(__EXCLUDE_LOGS__)
#define TRACCC_DEBUG_HOST_DEVICE(x, ...) \
    DETRAY_LOG_PRINTF("TRACCC", "DEBUG", x, __VA_ARGS__)
#else
#define TRACCC_DEBUG_HOST_DEVICE(x, ...)
#endif

// Device log messages in device/host-device code
#if defined(__DEVICE_LOGGING__) && !defined(__EXCLUDE_LOGS__)
#define TRACCC_WARNING_DEVICE(x, ...) TRACCC_WARNING_HOST_DEVICE(x, __VA_ARGS__)
#define TRACCC_ERROR_DEVICE(x, ...) TRACCC_ERROR_HOST_DEVICE(x, __VA_ARGS__)
#define TRACCC_FATAL_DEVICE(x, ...) TRACCC_FATAL_HOST_DEVICE(x, __VA_ARGS__)
#define TRACCC_INFO_DEVICE(x, ...) TRACCC_INFO_HOST_DEVICE(x, __VA_ARGS__)
#define TRACCC_VERBOSE_DEVICE(x, ...) TRACCC_VERBOSE_HOST_DEVICE(x, __VA_ARGS__)
#define TRACCC_DEBUG_DEVICE(x, ...) TRACCC_DEBUG_HOST_DEVICE(x, __VA_ARGS__)
#else
#define TRACCC_WARNING_DEVICE(x, ...)
#define TRACCC_ERROR_DEVICE(x, ...)
#define TRACCC_FATAL_DEVICE(x, ...)
#define TRACCC_INFO_DEVICE(x, ...)
#define TRACCC_VERBOSE_DEVICE(x, ...)
#define TRACCC_DEBUG_DEVICE(x, ...)
#endif
