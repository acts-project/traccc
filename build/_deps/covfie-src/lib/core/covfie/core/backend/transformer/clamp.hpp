/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <algorithm>
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
struct clamp {
    using this_t = clamp<_backend_t>;
    static constexpr bool is_initial = false;

    using backend_t = _backend_t;

    using contravariant_input_t = typename backend_t::contravariant_input_t;
    using covariant_output_t = typename backend_t::covariant_output_t;

    struct configuration_t {
        typename contravariant_input_t::vector_t min, max;
    };

    static constexpr uint32_t IO_MAGIC_HEADER = 0xAB020002;

    struct owning_data_t {
        using parent_t = this_t;

        owning_data_t() = default;

        template <typename... Args>
        explicit owning_data_t(parameter_pack<configuration_t, Args...> && args)
            : m_min(args.x.min)
            , m_max(args.x.max)
            , m_backend(std::move(args.xs))
        {
        }

        explicit owning_data_t(
            const configuration_t & c, typename backend_t::owning_data_t && b
        )
            : m_min(c.min)
            , m_max(c.max)
            , m_backend(std::forward<typename backend_t::owning_data_t>(b))
        {
        }

        explicit owning_data_t(parameter_pack<owning_data_t> && conf)
            : owning_data_t(std::move(conf.x))
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
            return {m_min, m_max};
        }

        static owning_data_t read_binary(std::istream & fs)
        {
            utility::read_io_header(fs, IO_MAGIC_HEADER);

            auto min = utility::read_binary<decltype(m_min)>(fs);
            auto max = utility::read_binary<decltype(m_max)>(fs);
            typename backend_t::owning_data_t be =
                backend_t::owning_data_t::read_binary(fs);

            utility::read_io_footer(fs, IO_MAGIC_HEADER);

            return owning_data_t(configuration_t{min, max}, std::move(be));
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

            backend_t::owning_data_t::write_binary(fs, o.m_backend);

            utility::write_io_footer(fs, IO_MAGIC_HEADER);
        }

        typename contravariant_input_t::vector_t m_min, m_max;
        typename backend_t::owning_data_t m_backend;
    };

    struct non_owning_data_t {
        using parent_t = this_t;

        non_owning_data_t(const owning_data_t & src)
            : m_min(src.m_min)
            , m_max(src.m_max)
            , m_backend(src.m_backend)
        {
        }

        template <std::size_t... Is>
        COVFIE_DEVICE typename contravariant_input_t::vector_t
        adjust(typename contravariant_input_t::vector_t coord, std::index_sequence<Is...>)
            const
        {
            return {std::clamp(coord[Is], m_min[Is], m_max[Is])...};
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t coord) const
        {
            return m_backend.at(adjust(
                coord,
                std::make_index_sequence<contravariant_input_t::dimensions>{}
            ));
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
        typename backend_t::non_owning_data_t m_backend;
    };
};
}
