/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {
/**
 * @defgroup Track finding return type specifiers
 *
 * Optional parameters to track finding algorithms which instruct the
 * algorithm to return either unfitted tracks or fitted tracks directly.
 *
 * @{
 * @brief Return tracks with fitted track states.
 */
struct finding_return_fitted {};
/*
 * @brief Return tracks with unfitted track states.
 */
struct finding_return_unfitted {};
/*
 * @}
 */
}  // namespace traccc::device
