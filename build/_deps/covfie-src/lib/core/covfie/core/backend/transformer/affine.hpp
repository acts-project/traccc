/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstddef>
#include <iostream>

#include <covfie/core/algebra/affine.hpp>
#include <covfie/core/algebra/vector.hpp>
#include <covfie/core/concepts.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/utility/binary_io.hpp>

namespace covfie::backend {
template <concepts::field_backend _backend_t>
struct affine {
    using this_t = affine<_backend_t>;
    static constexpr bool is_initial = false;

    using backend_t = _backend_t;

    using contravariant_input_t = typename backend_t::contravariant_input_t;
    using contravariant_output_t = typename backend_t::contravariant_input_t;
    using covariant_input_t = typename backend_t::covariant_output_t;
    using covariant_output_t = typename backend_t::covariant_output_t;

    using matrix_t = algebra::affine<
        contravariant_input_t::dimensions,
        typename contravariant_input_t::scalar_t>;

    struct owning_data_t;

    using configuration_t = matrix_t;

    static constexpr uint32_t IO_MAGIC_HEADER = 0xAB020000;

    struct owning_data_t {
        using parent_t = this_t;

        owning_data_t() = default;
        owning_data_t(const owning_data_t &) = default;
        owning_data_t(owning_data_t &&) = default;
        owning_data_t & operator=(const owning_data_t &) = default;
        owning_data_t & operator=(owning_data_t &&) = default;

        template <typename... Args>
        explicit owning_data_t(parameter_pack<configuration_t, Args...> && args)
            : m_transform(args.x)
            , m_backend(std::move(args.xs))
        {
        }

        explicit owning_data_t(parameter_pack<owning_data_t> && conf)
            : owning_data_t(std::move(conf.x))
        {
        }

        template <
            typename T,
            std::enable_if_t<
                std::is_same_v<
                    typename T::parent_t::configuration_t,
                    configuration_t>,
                bool> = true>
        explicit owning_data_t(const T & o)
            : m_transform(o.m_transform)
            , m_backend(o.m_backend)
        {
        }

        explicit owning_data_t(
            const configuration_t & c, typename backend_t::owning_data_t && b
        )
            : m_transform(c)
            , m_backend(std::forward<typename backend_t::owning_data_t>(b))
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
            return m_transform;
        }

        static owning_data_t read_binary(std::istream & fs)
        {
            utility::read_io_header(fs, IO_MAGIC_HEADER);

            configuration_t trans = utility::read_binary<configuration_t>(fs);
            typename backend_t::owning_data_t be =
                backend_t::owning_data_t::read_binary(fs);

            utility::read_io_footer(fs, IO_MAGIC_HEADER);

            return owning_data_t(trans, std::move(be));
        }

        static void write_binary(std::ostream & fs, const owning_data_t & o)
        {
            utility::write_io_header(fs, IO_MAGIC_HEADER);

            fs.write(
                reinterpret_cast<const char *>(&o.m_transform),
                sizeof(decltype(o.m_transform))
            );

            backend_t::owning_data_t::write_binary(fs, o.m_backend);

            utility::write_io_footer(fs, IO_MAGIC_HEADER);
        }

        matrix_t m_transform;
        typename backend_t::owning_data_t m_backend;
    };

    struct non_owning_data_t {
        using parent_t = this_t;

        non_owning_data_t(const owning_data_t & src)
            : m_transform(src.m_transform)
            , m_backend(src.m_backend)
        {
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t c) const
        {
            covfie::algebra::vector<
                contravariant_input_t::dimensions,
                typename contravariant_input_t::scalar_t>
                v;

            for (std::size_t i = 0; i < contravariant_output_t::dimensions; ++i)
            {
                v(i) = c[i];
            }

            covfie::algebra::vector<
                contravariant_input_t::dimensions,
                typename contravariant_input_t::scalar_t>
                nv = m_transform * v;

            typename contravariant_output_t::vector_t nc;

            for (std::size_t i = 0; i < contravariant_output_t::dimensions; ++i)
            {
                nc[i] = nv(i);
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

        matrix_t m_transform;
        typename backend_t::non_owning_data_t m_backend;
    };
};
}
