/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// GTest include(s)
#include <gtest/gtest.h>

// System include(s)
#include <source_location>

namespace detray::detail {

template <template <typename> class check_t, typename detector_t,
          typename config_t = typename check_t<detector_t>::config>
void register_checks(const detector_t &det,
                     const typename detector_t::name_map &vol_names,
                     const config_t &cfg = {},
                     const typename detector_t::geometry_context gctx = {}) {

    const char *test_name = cfg.name().c_str();
    if (!test_name) {
        throw std::invalid_argument("Invalid test name");
    }

    const std::source_location src_loc{};

    ::testing::RegisterTest("detray_validation", test_name, nullptr, test_name,
                            src_loc.file_name(), src_loc.line(),
                            [&det, &vol_names, &cfg, &gctx]() ->
                            typename check_t<detector_t>::fixture_type * {
                                return new check_t<detector_t>(det, vol_names,
                                                               cfg, gctx);
                            });
}

}  // namespace detray::detail
