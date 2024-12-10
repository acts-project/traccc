/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <iostream>
#include <variant>

#include <covfie/core/concepts.hpp>
#include <covfie/core/definitions.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/utility/binary_io.hpp>

namespace covfie::backend {
template <concepts::field_backend _backend_t, typename _shuffle>
struct shuffle {
    using this_t = shuffle<_backend_t, _shuffle>;
    static constexpr bool is_initial = false;

    using backend_t = _backend_t;

    using contravariant_input_t = typename backend_t::contravariant_input_t;
    using covariant_output_t = typename backend_t::covariant_output_t;

    using configuration_t = std::monostate;
    using indices = _shuffle;

    static constexpr uint32_t IO_MAGIC_HEADER = 0xAB020009;

    struct owning_data_t {
        using parent_t = this_t;

        owning_data_t() = default;

        template <typename... Args>
        explicit owning_data_t(configuration_t, Args... args)
            : m_backend(std::forward<Args>(args)...)
        {
        }

        explicit owning_data_t(
            const configuration_t &, typename backend_t::owning_data_t && b
        )
            : m_backend(std::forward<typename backend_t::owning_data_t>(b))
        {
        }

        template <typename... Args>
        explicit owning_data_t(parameter_pack<configuration_t, Args...> && args)
            : m_backend(std::move(args.xs))
        {
        }

        explicit owning_data_t(parameter_pack<owning_data_t> && conf)
            : owning_data_t(std::move(conf.x))
        {
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
            return {};
        }

        static owning_data_t read_binary(std::istream & fs)
        {
            auto be = backend_t::owning_data_t::read_binary(fs);

            return owning_data_t(configuration_t{}, std::move(be));
        }

        static void write_binary(std::ostream & fs, const owning_data_t & o)
        {
            backend_t::owning_data_t::write_binary(fs, o.m_backend);
        }

        typename backend_t::owning_data_t m_backend;
    };

    struct non_owning_data_t {
        using parent_t = this_t;

        non_owning_data_t(const owning_data_t & src)
            : m_backend(src.m_backend)
        {
        }

        template <std::size_t... Is>
        COVFIE_DEVICE typename contravariant_input_t::vector_t
        shuffle(typename contravariant_input_t::vector_t c, std::index_sequence<Is...>)
            const
        {
            return {c.at(Is)...};
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t c) const
        {
            return m_backend.at(shuffle(c, indices{}));
        }

        typename backend_t::non_owning_data_t & get_backend(void)
        {
            return m_backend;
        }

        const typename backend_t::non_owning_data_t & get_backend(void) const
        {
            return m_backend;
        }

        typename backend_t::non_owning_data_t m_backend;
    };
};
}
