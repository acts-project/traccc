/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "make_jagged_vector.hpp"

namespace vecmem::benchmark {

jagged_vector<int> make_jagged_vector(const std::vector<std::size_t>& sizes,
                                      memory_resource& mr) {

    // Create the result object.
    jagged_vector<int> result(&mr);
    result.reserve(sizes.size());
    for (std::size_t size : sizes) {
        result.push_back(jagged_vector<int>::value_type(size, &mr));
    }

    // Return the vector.
    return result;
}

}  // namespace vecmem::benchmark
