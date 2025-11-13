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
    SUCCESS,
    ERROR_QOP_ZERO,
    ERROR_THETA_POLE,
    ERROR_INVERSION,
    ERROR_SMOOTHER_CHI2_NEGATIVE,
    ERROR_SMOOTHER_CHI2_NOT_FINITE,
    ERROR_SMOOTHER_INVALID_COVARIANCE,
    ERROR_UPDATER_CHI2_NEGATIVE,
    ERROR_UPDATER_CHI2_NOT_FINITE,
    ERROR_UPDATER_INVALID_COVARIANCE,
    ERROR_BARCODE_SEQUENCE_OVERFLOW,
    ERROR_INVALID_TRACK_STATE,
    ERROR_TRACK_STATES_EMPTY,
    ERROR_NOT_ALL_TRACK_STATES_FOUND,
    ERROR_OTHER,
    MAX_STATUS
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
            case ERROR_UPDATER_CHI2_NEGATIVE: {
                return msg + "Negative chi2 in gain matrix updater";
            }
            case ERROR_UPDATER_CHI2_NOT_FINITE: {
                return msg + "Invalid chi2 in gain matrix updater";
            }
            case ERROR_UPDATER_INVALID_COVARIANCE: {
                return msg + "Invalid track covariance in forward fit";
            }
            case ERROR_BARCODE_SEQUENCE_OVERFLOW: {
                return msg + "Barcode sequence overflow in direct navigator";
            }
            case ERROR_INVALID_TRACK_STATE: {
                return msg +
                       "Invalid track state in forward pass (skipped or error)";
            }
            case ERROR_TRACK_STATES_EMPTY: {
                return msg + "List of track states was empty";
            }
            case ERROR_NOT_ALL_TRACK_STATES_FOUND: {
                return msg + "Not all track states were recovered in fit";
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
