/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cstdint>
namespace traccc {
enum class kalman_fitter_status : uint32_t {
    SUCCESS,
    ERROR_QOP_ZERO,
    ERROR_THETA_ZERO,
    ERROR_INVERSION,
    ERROR_SMOOTHER_CHI2_NEGATIVE,
    ERROR_UPDATER_CHI2_NEGATIVE,
    ERROR_OTHER,
    MAX_STATUS
};
}  // namespace traccc
