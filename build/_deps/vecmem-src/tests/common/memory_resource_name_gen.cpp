/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "memory_resource_name_gen.hpp"

namespace vecmem::testing {

memory_resource_name_gen::memory_resource_name_gen(const storage_type& names)
    : m_names(names), m_unknown_count(0) {}

std::string memory_resource_name_gen::operator()(
    const ::testing::TestParamInfo<memory_resource*>& info) {

    // Look for the memory resource amongst the known ones.
    auto itr = m_names.find(info.param);
    if (itr != m_names.end()) {
        return itr->second;
    }

    // If it's not a known type, give it a name now.
    const std::string unknown_name =
        "Unknown" + std::to_string(++m_unknown_count);
    m_names[info.param] = unknown_name;
    return unknown_name;
}

}  // namespace vecmem::testing
