/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// This header is used to disable GCC errors that could be occuring in
// nlohmann/json.hpp
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#include <nlohmann/json.hpp>
#pragma GCC diagnostic pop
