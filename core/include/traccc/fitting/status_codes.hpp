/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cstdint>
#include <string>

namespace traccc {
enum class kalman_fitter_status : uint32_t {
    SUCCESS = 0u,
    ERROR_QOP_ZERO = 1u,
    ERROR_THETA_POLE = 2u,
    ERROR_INVERSION = 3u,
    ERROR_SMOOTHER_CHI2_NEGATIVE = 4u,
    ERROR_SMOOTHER_CHI2_NOT_FINITE = 5u,
    ERROR_SMOOTHER_INVALID_COVARIANCE = 6u,
    ERROR_SMOOTHER_SKIPPED_STATE = 7u,
    ERROR_UPDATER_CHI2_NEGATIVE = 8u,
    ERROR_UPDATER_CHI2_NOT_FINITE = 9u,
    ERROR_UPDATER_INVALID_COVARIANCE = 10u,
    ERROR_UPDATER_SKIPPED_STATE = 11u,
    ERROR_BARCODE_SEQUENCE_OVERFLOW = 12u,
    ERROR_PROPAGATION_FAILURE = 13u,
    ERROR_OTHER = 14u,
    MAX_STATUS = 15u
};

/// Convert a status code into a human readable string
struct fitter_debug_msg {

    TRACCC_HOST std::string operator()() const {
        const std::string msg{"Kalman Fitter: "};
        switch (m_error_code) {
            using enum kalman_fitter_status;
            case ERROR_QOP_ZERO: {
                return msg + "Track qop is zero";
            }
            case ERROR_THETA_POLE: {
                return msg + "Track theta hit pole";
            }
            case ERROR_INVERSION: {
                return msg + "Failed matrix inversion";
            }
            case ERROR_SMOOTHER_CHI2_NEGATIVE: {
                return msg + "Negative chi2 in smoother";
            }
            case ERROR_SMOOTHER_CHI2_NOT_FINITE: {
                return msg + "Invalid chi2 in smoother";
            }
            case ERROR_SMOOTHER_INVALID_COVARIANCE: {
                return msg + "Invalid track covariance during smoothing";
            }
            case ERROR_SMOOTHER_SKIPPED_STATE: {
                return msg + "Skipped track state during smoothing";
            }
            case ERROR_UPDATER_CHI2_NEGATIVE: {
                return msg + "Negative chi2 in gain matrix updater";
            }
            case ERROR_UPDATER_CHI2_NOT_FINITE: {
                return msg + "Invalid chi2 in gain matrix updater";
            }
            case ERROR_UPDATER_INVALID_COVARIANCE: {
                return msg + "Invalid track covariance in forward fit";
            }
            case ERROR_UPDATER_SKIPPED_STATE: {
                return msg + "Skipped track state during forward fit";
            }
            case ERROR_BARCODE_SEQUENCE_OVERFLOW: {
                return msg + "Barcode sequence overflow in direct navigator";
            }
            case ERROR_PROPAGATION_FAILURE: {
                return msg + "Propagation failure";
            }
            case ERROR_OTHER: {
                return msg + "Unspecified error";
            }
            default: {
                return "Not a defined error code: " +
                       std::to_string(static_cast<int>(m_error_code));
            }
        }
    }

    kalman_fitter_status m_error_code{kalman_fitter_status::MAX_STATUS};
};

}  // namespace traccc
