/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once
#include <sstream>
#include <stdexcept>
#include <string>
#include <traccc/futhark/context.hpp>

#define FUTHARK_ERROR_CHECK(ans) futharkAssert((ans), __FILE__, __LINE__)

inline void futharkAssert(int code, const char *file, int line,
                          bool abort = true) {
    char *error = futhark_context_get_error(&traccc::futhark::get_context());

    if (code == 2) {
        throw std::runtime_error(
            "Futhark program exited due to a programming error.");
    } else if (code == 3) {
        throw std::runtime_error(
            "Futhark program exited due to lack of allocatable memory.");
    } else if (code != 0) {
        std::stringstream ss;
        ss << "Futhark program exited with unknown non-zero return code "
           << code << ".";

        if (error != nullptr) {
            ss << " Additional error informtion: " << std::string(error);
        }

        throw std::runtime_error(ss.str());
    }
}
