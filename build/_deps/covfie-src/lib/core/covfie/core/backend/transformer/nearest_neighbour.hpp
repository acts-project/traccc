/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <variant>

#include <covfie/core/concepts.hpp>
#include <covfie/core/definitions.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/utility/binary_io.hpp>
#include <covfie/core/vector.hpp>

namespace covfie::backend {
template <
    concepts::field_backend _backend_t,
    concepts::vector_descriptor _input_vector_d = covfie::vector::
        vector_d<float, _backend_t::contravariant_input_t::dimensions>>
struct nearest_neighbour {
    using this_t = nearest_neighbour<_backend_t, _input_vector_d>;
    static constexpr bool is_initial = false;

    using backend_t = _backend_t;

    static_assert(
        std::is_floating_point_v<typename _input_vector_d::type>,
        "Nearest neighbour interpolation contravariant input must have a "
        "floating point scalar type."
    );
    static_assert(
        _input_vector_d::size == backend_t::contravariant_input_t::dimensions,
        "Nearest neighbour interpolation contravariant input must have the "
        "same size as the backend contravariant input."
    );

    using contravariant_input_t =
        covfie::vector::array_vector_d<_input_vector_d>;
    using contravariant_output_t = typename backend_t::contravariant_input_t;
    using covariant_input_t = typename backend_t::covariant_output_t;
    using covariant_output_t = covariant_input_t;

    using configuration_t = std::monostate;

    static constexpr uint32_t IO_MAGIC_HEADER = 0xAB020007;

    struct owning_data_t {
        using parent_t = this_t;

        owning_data_t() = default;
        owning_data_t(const owning_data_t &) = default;
        owning_data_t(owning_data_t &&) = default;
        owning_data_t & operator=(const owning_data_t &) = default;
        owning_data_t & operator=(owning_data_t &&) = default;

        template <typename... Args>
        explicit owning_data_t(parameter_pack<configuration_t, Args...> && args)
            : m_backend(std::move(args.xs))
        {
        }

        explicit owning_data_t(parameter_pack<owning_data_t> && conf)
            : owning_data_t(std::move(conf.x))
        {
        }

        template <
            typename T,
            typename... Args,
            std::enable_if_t<
                std::is_same_v<
                    typename T::parent_t::configuration_t,
                    std::monostate>,
                bool> = true>
        explicit owning_data_t(const T & o)
            : m_backend(o.get_backend())
        {
        }

        explicit owning_data_t(
            const configuration_t &, typename backend_t::owning_data_t && b
        )
            : m_backend(std::forward<typename backend_t::owning_data_t>(b))
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

        COVFIE_DEVICE typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t c) const
        {
            typename contravariant_output_t::vector_t nc;

            for (std::size_t i = 0; i < contravariant_output_t::dimensions; ++i)
            {
                nc[i] = static_cast<typename contravariant_output_t::scalar_t>(
                    std::lrintf(c[i])
                );
            }

            return m_backend.at(nc);
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
