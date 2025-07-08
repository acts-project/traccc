/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "make_magnetic_field.hpp"

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/io/read_bfield.hpp"

namespace traccc::details {

bfield make_magnetic_field(const opts::magnetic_field& opts) {

    if (opts.read_from_file) {
        covfie::field<inhom_bfield_backend_t<scalar>> field;
        io::read_bfield(field, opts.file, opts.format);
        return bfield{std::move(field)};
    } else {
        return bfield{construct_const_bfield<scalar>({0.f, 0.f, opts.value})};
    }
}

}  // namespace traccc::details
