/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "bfield.cuh"
#include "traccc/cuda/utils/make_bfield.hpp"

// Project include(s).
#include "traccc/definitions/common.hpp"

// System include(s).
#include <stdexcept>

namespace traccc::cuda {

bfield make_bfield(const bfield& field) {

    if (field.is<const_bfield_backend_t<scalar>>()) {
        return bfield{covfie::field<const_bfield_backend_t<scalar>>{
            field.get_covfie_field<const_bfield_backend_t<scalar>>()}};
    } else if (field.is<traccc::inhom_bfield_backend_t<scalar>>()) {
        return bfield{covfie::field<cuda::inhom_bfield_backend_t<scalar>>(
            field.get_covfie_field<traccc::inhom_bfield_backend_t<scalar>>())};
    } else {
        throw std::invalid_argument("Unsupported b-field type received");
    }
}

}  // namespace traccc::cuda
