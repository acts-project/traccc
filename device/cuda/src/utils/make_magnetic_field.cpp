/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/cuda/utils/make_magnetic_field.hpp"

#include "magnetic_field_types.hpp"

// Project include(s).
#include "traccc/bfield/magnetic_field_types.hpp"
#include "traccc/definitions/common.hpp"

// System include(s).
#include <stdexcept>

namespace traccc::cuda {

magnetic_field make_magnetic_field(const magnetic_field& bfield) {

    if (bfield.is<const_bfield_backend_t<scalar>>()) {
        return magnetic_field{covfie::field<const_bfield_backend_t<scalar>>{
            bfield.as_field<const_bfield_backend_t<scalar>>()}};
    } else if (bfield.is<host::inhom_bfield_backend_t<scalar>>()) {
        return magnetic_field{
            covfie::field<cuda::inhom_bfield_backend_t<scalar>>(
                bfield.as_field<host::inhom_bfield_backend_t<scalar>>())};
    } else {
        throw std::invalid_argument("Unsupported b-field type received");
    }
}

}  // namespace traccc::cuda
