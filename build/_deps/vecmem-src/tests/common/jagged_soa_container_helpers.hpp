/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "jagged_soa_container.hpp"

namespace vecmem {
namespace testing {

/// Fill a device container with some dummy data
VECMEM_HOST_AND_DEVICE
inline void fill(unsigned int i, jagged_soa_container::device& obj) {

    // In the first thread modify the scalars.
    if (i == 0) {
        obj.count() = 55;
        obj.average() = 3.141592f;
    }
    // In the rest of the threads modify the vector variables.
    if (i < obj.size()) {
        obj.measurement()[i] = 1.0f * static_cast<float>(i);
        obj.index()[i] = static_cast<int>(i);
        for (unsigned int j = 0; j < obj.measurements()[i].capacity(); ++j) {
            obj.measurements()[i].push_back(1.f * static_cast<double>(i + j));
        }
        for (unsigned int j = 0; j < obj.indices()[i].capacity(); ++j) {
            obj[i].indices().push_back(static_cast<int>(i + j));
        }
    }
}

/// Modify the contents of a device container
VECMEM_HOST_AND_DEVICE
inline void modify(unsigned int i, jagged_soa_container::device& obj) {

    // In the first thread modify the scalars.
    if (i == 0) {
        obj.count() += 2;
        obj.average() -= 1.f;
    }
    // In the rest of the threads modify the vector variables.
    if (i < obj.size()) {
        obj.measurement()[i] *= 2.f;
        obj.index()[i] += 10;
        for (unsigned int j = 0; j < obj.measurements()[i].size(); ++j) {
            obj.at(i).measurements()[j] *= 2.;
        }
        for (unsigned int j = 0; j < obj.indices()[i].size(); ++j) {
            obj.indices()[i][j] += 10;
        }
    }
}

/// Helper function testing the equality of two containers
void compare(const jagged_soa_container::const_view& lhs,
             const jagged_soa_container::const_view& rhs);

#if __cplusplus >= 201700L

/// Fill a host container with some dummy data
void fill(jagged_soa_container::host& obj);

/// Create a buffer for the tests
void make_buffer(jagged_soa_container::buffer& buffer, memory_resource& main_mr,
                 memory_resource& host_mr, data::buffer_type buffer_type);

#endif

}  // namespace testing
}  // namespace vecmem
