/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem {

template <typename TYPE, typename ALLOC>
VECMEM_HOST data::vector_view<TYPE> get_data(std::vector<TYPE, ALLOC>& vec) {

    return {
        static_cast<typename data::vector_view<TYPE>::size_type>(vec.size()),
        vec.data()};
}

template <typename TYPE, typename ALLOC>
VECMEM_HOST data::vector_view<const TYPE> get_data(
    const std::vector<TYPE, ALLOC>& vec) {

    return {static_cast<typename data::vector_view<const TYPE>::size_type>(
                vec.size()),
            vec.data()};
}

}  // namespace vecmem
