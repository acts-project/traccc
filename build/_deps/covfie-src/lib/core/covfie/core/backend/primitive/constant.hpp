/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <iostream>

#include <covfie/core/concepts.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/utility/binary_io.hpp>
#include <covfie/core/vector.hpp>

namespace covfie::backend {
template <
    concepts::vector_descriptor _input_vector_t,
    concepts::vector_descriptor _output_vector_t>
struct constant {
    using this_t = constant<_input_vector_t, _output_vector_t>;
    static constexpr bool is_initial = true;

    using contravariant_input_t =
        typename covfie::vector::array_vector_d<_input_vector_t>;
    using covariant_output_t =
        typename covfie::vector::array_vector_d<_output_vector_t>;

    struct owning_data_t;

    using configuration_t = typename covariant_output_t::vector_t;

    static constexpr uint32_t IO_MAGIC_HEADER = 0xAB010001;

    struct owning_data_t {
        using parent_t = this_t;

        owning_data_t() = default;
        owning_data_t(const owning_data_t &) = default;
        owning_data_t & operator=(const owning_data_t &) = default;
        owning_data_t & operator=(owning_data_t &&) = default;

        explicit owning_data_t(configuration_t conf)
            : m_value(conf)
        {
        }

        explicit owning_data_t(parameter_pack<configuration_t> && conf)
            : owning_data_t(std::move(conf.x))
        {
        }

        explicit owning_data_t(parameter_pack<owning_data_t> && conf)
            : m_value(std::move(conf.x.m_value))
        {
        }

        configuration_t get_configuration() const
        {
            return m_value;
        }

        static owning_data_t read_binary(std::istream & fs)
        {
            utility::read_io_header(fs, IO_MAGIC_HEADER);

            auto vec =
                utility::read_binary<typename covariant_output_t::vector_t>();

            utility::read_io_footer(fs, IO_MAGIC_HEADER);

            return owning_data_t(vec);
        }

        static void write_binary(std::ostream & fs, const owning_data_t & o)
        {
            utility::write_io_header(fs, IO_MAGIC_HEADER);

            fs.write(
                reinterpret_cast<const char *>(&o.m_value),
                sizeof(decltype(o.m_value))
            );

            utility::write_io_footer(fs, IO_MAGIC_HEADER);
        }

        typename covariant_output_t::vector_t m_value;
    };

    struct non_owning_data_t {
        using parent_t = this_t;

        non_owning_data_t(const owning_data_t & o)
            : m_value(o.m_value)
        {
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t
            at(typename contravariant_input_t::vector_t) const
        {
            return m_value;
        }

        typename covariant_output_t::vector_t m_value;
    };
};
}
