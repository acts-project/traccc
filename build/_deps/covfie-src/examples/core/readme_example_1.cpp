/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <cmath>
#include <iostream>

#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/field_view.hpp>
#include <covfie/core/parameter_pack.hpp>

using field_t = covfie::field<covfie::backend::strided<
    covfie::vector::size2,
    covfie::backend::array<covfie::vector::float2>>>;

int main(void)
{
    // Initialize the field as a 10x10 field, then create a view from it.
    field_t my_field(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t{10ul, 10ul}
    ));
    field_t::view_t my_view(my_field);

    // Assign f(x, y) = (sin x, cos y)
    for (std::size_t x = 0ul; x < 10ul; ++x) {
        for (std::size_t y = 0ul; y < 10ul; ++y) {
            my_view.at(x, y)[0] = std::sin(static_cast<float>(x));
            my_view.at(x, y)[1] = std::cos(static_cast<float>(y));
        }
    }

    // Retrieve the vector value at (2, 3)
    field_t::output_t v = my_view.at(2ul, 3ul);

    std::cout << "Value at (2, 3) = (" << v[0] << ", " << v[1] << ")"
              << std::endl;

    return 0;
}
