/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/performance/timer.hpp"

// System include(s).
#include <algorithm>

// Nividia tool extensions library
#ifdef TRACCC_HAVE_NVTX
  #include "nvtx3/nvToolsExt.h"
#endif //TRACCC_HAVE_NVTX

namespace traccc::performance {

/// Start time measurement
/// @param timer_name name to be printed out identifying what is measured
/// @param t_info shared_ptr to timing_info where to store timings
timer::timer(const std::string_view timer_name, timing_info& t_info)
    : m_start(std::chrono::high_resolution_clock::now()),
      m_name(timer_name),
      m_timing_info(t_info) {
      nvtxRangePushA(timer_name.data());
      }

/// End time measurement
timer::~timer() {
    nvtxRangePop();
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::nanoseconds totalTime = end - m_start;
    const auto pos =
        std::find_if(m_timing_info.data.begin(), m_timing_info.data.end(),
                     [&m_name = m_name](const timing_info_pair& element) {
                         return element.first == m_name;
                     });

    if (pos == m_timing_info.data.end()) {
        m_timing_info.data.push_back({m_name, totalTime});
    } else {
        pos->second += totalTime;
    }

}

}  // namespace traccc::performance
