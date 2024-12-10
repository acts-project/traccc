/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <iostream>
#include <type_traits>
#include <variant>

#include <covfie/core/concepts.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/core/utility/binary_io.hpp>
#include <covfie/core/vector.hpp>

namespace covfie::backend {
template <concepts::vector_descriptor _vector_t>
struct identity {
    using this_t = identity<_vector_t>;
    static constexpr bool is_initial = true;

    using contravariant_input_t = covfie::vector::array_vector_d<_vector_t>;
    using covariant_output_t = covfie::vector::array_vector_d<_vector_t>;

    static_assert(
        contravariant_input_t::dimensions == covariant_output_t::dimensions,
        "Identity backend requires input and output to have identical "
        "dimensionality."
    );
    static_assert(
        std::is_constructible_v<
            typename contravariant_input_t::scalar_t,
            typename covariant_output_t::scalar_t>,
        "Identity backend requires type of input to be convertible to type of "
        "output."
    );

    struct owning_data_t;

    using configuration_t = std::monostate;

    static constexpr uint32_t IO_MAGIC_HEADER = 0xAB010002;

    struct owning_data_t {
        using parent_t = this_t;

        explicit owning_data_t()
        {
        }

        explicit owning_data_t(configuration_t)
        {
        }

        explicit owning_data_t(parameter_pack<configuration_t> &&)
        {
        }

        explicit owning_data_t(parameter_pack<owning_data_t> &&)
        {
        }

        configuration_t get_configuration() const
        {
            return {};
        }

        static owning_data_t read_binary(std::istream & fs)
        {
            utility::read_io_header(fs, IO_MAGIC_HEADER);
            utility::read_io_footer(fs, IO_MAGIC_HEADER);

            return owning_data_t();
        }

        static void write_binary(std::ostream & fs, const owning_data_t &)
        {
            utility::write_io_header(fs, IO_MAGIC_HEADER);
            utility::write_io_footer(fs, IO_MAGIC_HEADER);
        }
    };

    struct non_owning_data_t {
        using parent_t = this_t;

        non_owning_data_t(const owning_data_t &)
        {
        }

        typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t c) const
        {
            typename covariant_output_t::vector_t rv;

            for (std::size_t i = 0ul; i < contravariant_input_t::dimensions;
                 ++i)
            {
                rv[i] = c[i];
            }

            return rv;
        }
    };
};
}
