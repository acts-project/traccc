/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <iostream>

#if !defined(TRACCC_BUILD_TYPE_IS_RELEASE) || !defined(NDEBUG) || \
    defined(_DEBUG)
#define TRACCC_OPTIMIZATION_WARNING()                                         \
    do {                                                                      \
        std::cout                                                             \
            << "WARNING: traccc was built without Release mode, without the " \
               "`NDEBUG` flag, or (on MSVC) with the `_DEBUG` flag. "         \
               "Performance is guaranteed to be much lower and compute "      \
               "performance results should be considered unreliable!"         \
            << std::endl;                                                     \
    } while (false)
#else
#define TRACCC_OPTIMIZATION_WARNING() \
    do {                              \
    } while (false)
#endif
