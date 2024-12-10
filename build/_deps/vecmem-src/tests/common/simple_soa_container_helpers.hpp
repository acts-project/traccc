/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "simple_soa_container.hpp"

// Project include(s).
#include "vecmem/utils/types.hpp"

namespace vecmem {
namespace testing {

/// Fill a device container with some dummy data
VECMEM_HOST_AND_DEVICE
inline void fill(unsigned int i, simple_soa_container::device& obj) {

    // In the first thread modify the scalars.
    if (i == 0) {
        obj.count() = 55;
        obj.average() = 3.141592f;
    }
    // In the rest of the threads modify the vector variables.
    if (i < obj.capacity()) {
        unsigned int ii = obj.push_back_default();
        obj.measurement()[ii] = 1.0f * static_cast<float>(ii);
        obj.at(ii).index() = static_cast<int>(ii);
    }
}

/// Modify the contents of a device container
VECMEM_HOST_AND_DEVICE
inline void modify(unsigned int i, simple_soa_container::device& obj) {

    // In the first thread modify the scalars.
    if (i == 0) {
        obj.count() += 2;
        obj.average() -= 1.0f;
    }
    // In the rest of the threads modify the vector variables.
    if (i < obj.size()) {
        obj.at(i).measurement() *= 2.0f;
        obj.index()[i] += 10;
    }
}

/// Helper function testing the equality of two containers
void compare(const simple_soa_container::const_view& lhs,
             const simple_soa_container::const_view& rhs);

#if __cplusplus >= 201700L

/// Fill a host container with some dummy data
void fill(simple_soa_container::host& obj);

/// Create a buffer for the tests
void make_buffer(simple_soa_container::buffer& buffer, memory_resource& main_mr,
                 memory_resource& host_mr, data::buffer_type buffer_type);

#endif

}  // namespace testing
}  // namespace vecmem
