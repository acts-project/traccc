/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc {
namespace array {

template <typename T, std::size_t N>
using array = darray<T, N>;
using transform3 = algebra::array::transform3;

}  // namespace array

namespace eigen {

template <typename T, std::size_t N>
using array = Eigen::Matrix<T, N, 1>;
using transform3 = algebra::eigen::transform3;
    
}  // namespace eigen
    
}  // namespace traccc


