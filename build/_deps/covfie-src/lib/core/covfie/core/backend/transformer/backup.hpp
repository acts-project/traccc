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
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/utility/binary_io.hpp>
#include <covfie/core/utility/nd_size.hpp>
#include <covfie/core/vector.hpp>

namespace covfie::backend {
template <concepts::field_backend _backend_t>
struct backup {
    using this_t = backup<_backend_t>;
    static constexpr bool is_initial = false;

    using backend_t = _backend_t;

    using contravariant_input_t = typename backend_t::contravariant_input_t;
    using contravariant_output_t = typename backend_t::contravariant_input_t;
    using covariant_input_t = typename backend_t::covariant_output_t;
    using covariant_output_t =
        covfie::vector::array_vector_d<typename covariant_input_t::vector_d>;

    struct configuration_t {
        typename contravariant_input_t::vector_t min, max;
        typename covariant_output_t::vector_t default_value;
    };

    static constexpr uint32_t IO_MAGIC_HEADER = 0xAB020001;

    struct owning_data_t {
        using parent_t = this_t;

        owning_data_t() = default;
        owning_data_t(const owning_data_t &) = default;
        owning_data_t(owning_data_t &&) = default;
        owning_data_t & operator=(const owning_data_t &) = default;
        owning_data_t & operator=(owning_data_t &&) = default;

        template <typename... Args>
        explicit owning_data_t(configuration_t conf, Args... args)
            : m_min(conf.min)
            , m_max(conf.max)
            , m_default(conf.default_value)
            , m_backend(std::forward<Args>(args)...)
        {
        }

        template <typename... Args>
        explicit owning_data_t(parameter_pack<configuration_t, Args...> && args)
            : m_min(args.x.min)
            , m_max(args.x.max)
            , m_default(args.x.default_value)
            , m_backend(std::move(args.xs))
        {
        }

        explicit owning_data_t(parameter_pack<owning_data_t> && conf)
            : owning_data_t(std::move(conf.x))
        {
        }

        explicit owning_data_t(
            const configuration_t & c, typename backend_t::owning_data_t && b
        )
            : m_min(c.min)
            , m_max(c.max)
            , m_default(c.default_value)
            , m_backend(std::forward<typename backend_t::owning_data_t>(b))
        {
        }

        template <
            typename... Args,
            typename B = backend_t,
            std::enable_if_t<
                std::is_same_v<
                    typename B::configuration_t,
                    utility::nd_size<B::contravariant_input_t::dimensions>>,
                bool> = true>
        explicit owning_data_t(Args... args)
            : m_backend(std::forward<Args>(args)...)
        {
            m_min.fill(static_cast<typename contravariant_input_t::scalar_t>(0)
            );

            for (std::size_t i = 0; i < contravariant_input_t::dimensions; ++i)
            {
                m_max[i] = m_backend.get_configuration()[i];
            }

            m_default.fill(static_cast<typename covariant_output_t::scalar_t>(0)
            );
        }

        typename backend_t::owning_data_t & get_backend(void)
        {
            return m_backend;
        }

        const typename backend_t::owning_data_t & get_backend(void) const
        {
            return m_backend;
        }

        configuration_t get_configuration(void) const
        {
            return {m_min, m_max, m_default};
        }

        static owning_data_t read_binary(std::istream & fs)
        {
            utility::read_io_header(fs, IO_MAGIC_HEADER);

            auto min = utility::read_binary<decltype(m_min)>(fs);
            auto max = utility::read_binary<decltype(m_min)>(fs);
            auto def = utility::read_binary<decltype(m_default)>(fs);
            typename backend_t::owning_data_t be =
                backend_t::owning_data_t::read_binary(fs);

            utility::read_io_footer(fs, IO_MAGIC_HEADER);

            return owning_data_t(configuration_t{min, max, def}, std::move(be));
        }

        static void write_binary(std::ostream & fs, const owning_data_t & o)
        {
            utility::write_io_header(fs, IO_MAGIC_HEADER);

            fs.write(
                reinterpret_cast<const char *>(&o.m_min),
                sizeof(decltype(m_min))
            );
            fs.write(
                reinterpret_cast<const char *>(&o.m_max),
                sizeof(decltype(m_max))
            );
            fs.write(
                reinterpret_cast<const char *>(&o.m_default),
                sizeof(decltype(m_default))
            );

            backend_t::owning_data_t::write_binary(fs, o.m_backend);

            utility::write_io_footer(fs, IO_MAGIC_HEADER);
        }

        typename contravariant_input_t::vector_t m_min, m_max;
        typename covariant_output_t::vector_t m_default;
        typename backend_t::owning_data_t m_backend;
    };

    struct non_owning_data_t {
        using parent_t = this_t;

        non_owning_data_t(const owning_data_t & src)
            : m_min(src.m_min)
            , m_max(src.m_max)
            , m_default(src.m_default)
            , m_backend(src.m_backend)
        {
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t coord) const
        {
            for (std::size_t i = 0; i < contravariant_input_t::dimensions; ++i)
            {
                if (coord[i] < m_min[i] || coord[i] > m_max[i]) {
                    return m_default;
                }
            }

            return m_backend.at(coord);
        }

        typename backend_t::non_owning_data_t & get_backend(void)
        {
            return m_backend;
        }

        const typename backend_t::non_owning_data_t & get_backend(void) const
        {
            return m_backend;
        }

        typename contravariant_input_t::vector_t m_min, m_max;
        typename covariant_output_t::vector_t m_default;
        typename backend_t::non_owning_data_t m_backend;
    };
};
}
