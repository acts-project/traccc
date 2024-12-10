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
struct linear {
    using this_t = linear<_backend_t, _input_vector_d>;
    static constexpr bool is_initial = false;

    using input_scalar_type = typename _input_vector_d::type;
    using backend_t = _backend_t;

    using contravariant_input_t =
        covfie::vector::array_vector_d<_input_vector_d>;
    using contravariant_output_t = typename backend_t::contravariant_input_t;
    using covariant_input_t = typename backend_t::covariant_output_t;
    using covariant_output_t =
        covfie::vector::array_vector_d<typename covariant_input_t::vector_d>;

    static_assert(
        std::is_floating_point_v<typename _input_vector_d::type>,
        "Linear interpolation contravariant input must have a "
        "floating point scalar type."
    );
    static_assert(
        std::is_floating_point_v<typename covariant_input_t::scalar_t>,
        "Linear interpolation covariant input must have a "
        "floating point scalar type."
    );
    static_assert(
        _input_vector_d::size == backend_t::contravariant_input_t::dimensions,
        "Linear interpolation contravariant input must have the "
        "same size as the backend contravariant input."
    );
    static_assert(
        std::is_object_v<typename covariant_output_t::vector_t>,
        "Covariant input type of linear interpolator must be an object type."
    );

    using configuration_t = std::monostate;

    static constexpr uint32_t IO_MAGIC_HEADER = 0xAB020005;

    struct owning_data_t {
        using parent_t = this_t;

        owning_data_t() = default;

        template <typename... Args>
        explicit owning_data_t(configuration_t, Args... args)
            : m_backend(std::forward<Args>(args)...)
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
            : m_backend(o.m_backend)
        {
        }

        explicit owning_data_t(const typename backend_t::owning_data_t & o)
            : m_backend(o)
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
            auto be = decltype(m_backend)::read_binary(fs);

            return owning_data_t(configuration_t{}, std::move(be));
        }

        static void write_binary(std::ostream & fs, const owning_data_t & o)
        {
            decltype(m_backend)::write_binary(fs, o.m_backend);
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
        COVFIE_DEVICE typename contravariant_output_t::vector_t
        _backend_index_helper(typename contravariant_output_t::vector_t coord, std::size_t n, std::index_sequence<Is...>)
            const
        {
            return {static_cast<typename decltype(m_backend
            )::parent_t::contravariant_input_t::scalar_t>(
                coord[Is] + ((n & (std::size_t(1) << Is)) ? 1 : 0)
            )...};
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t coord) const
        {
            if constexpr (covariant_output_t::dimensions == 1) {
                typename contravariant_output_t::scalar_t i =
                    static_cast<typename contravariant_output_t::scalar_t>(
                        coord[0]
                    );

                input_scalar_type a = coord[0] - std::trunc(coord[0]);

                input_scalar_type ra = static_cast<input_scalar_type>(1.) - a;

                std::remove_reference_t<typename covariant_output_t::vector_t>
                    pc[2];

                for (std::size_t n = 0; n < 2; ++n) {
                    pc[n] =
                        m_backend.at({static_cast<typename decltype(m_backend
                        )::parent_t::contravariant_input_t::scalar_t>(
                            i + ((n & 1) ? 1 : 0)
                        )});
                }

                typename covariant_output_t::vector_t rv;

                for (std::size_t q = 0; q < covariant_output_t::dimensions; ++q)
                {
                    rv[q] = ra * static_cast<input_scalar_type>(pc[0][q]) +
                            a * static_cast<input_scalar_type>(pc[1][q]);
                }

                return rv;
            } else if constexpr (covariant_output_t::dimensions == 2) {
                typename contravariant_output_t::scalar_t i =
                    static_cast<typename contravariant_output_t::scalar_t>(
                        coord[0]
                    );
                typename contravariant_output_t::scalar_t j =
                    static_cast<typename contravariant_output_t::scalar_t>(
                        coord[1]
                    );

                input_scalar_type a = coord[0] - std::trunc(coord[0]);
                input_scalar_type b = coord[1] - std::trunc(coord[1]);

                input_scalar_type ra = static_cast<input_scalar_type>(1.) - a;
                input_scalar_type rb = static_cast<input_scalar_type>(1.) - b;

                std::remove_reference_t<typename covariant_output_t::vector_t>
                    pc[4];

                for (std::size_t n = 0; n < 4; ++n) {
                    pc[n] = m_backend.at(
                        {static_cast<typename decltype(m_backend
                         )::parent_t::contravariant_input_t::scalar_t>(
                             i + ((n & 2) ? 1 : 0)
                         ),
                         static_cast<typename decltype(m_backend
                         )::parent_t::contravariant_input_t::scalar_t>(
                             j + ((n & 1) ? 1 : 0)
                         )}
                    );
                }

                typename covariant_output_t::vector_t rv;

                for (std::size_t q = 0; q < covariant_output_t::dimensions; ++q)
                {
                    rv[q] = ra * rb * static_cast<input_scalar_type>(pc[0][q]) +
                            ra * b * static_cast<input_scalar_type>(pc[1][q]) +
                            a * rb * static_cast<input_scalar_type>(pc[2][q]) +
                            a * b * static_cast<input_scalar_type>(pc[3][q]);
                }

                return rv;
            } else if constexpr (covariant_output_t::dimensions == 3) {
                typename contravariant_output_t::scalar_t i =
                    static_cast<typename contravariant_output_t::scalar_t>(
                        coord[0]
                    );
                typename contravariant_output_t::scalar_t j =
                    static_cast<typename contravariant_output_t::scalar_t>(
                        coord[1]
                    );
                typename contravariant_output_t::scalar_t k =
                    static_cast<typename contravariant_output_t::scalar_t>(
                        coord[2]
                    );

                input_scalar_type a = coord[0] - std::trunc(coord[0]);
                input_scalar_type b = coord[1] - std::trunc(coord[1]);
                input_scalar_type c = coord[2] - std::trunc(coord[2]);

                input_scalar_type ra = static_cast<input_scalar_type>(1.) - a;
                input_scalar_type rb = static_cast<input_scalar_type>(1.) - b;
                input_scalar_type rc = static_cast<input_scalar_type>(1.) - c;

                std::remove_reference_t<typename covariant_output_t::vector_t>
                    pc[8];

                for (std::size_t n = 0; n < 8; ++n) {
                    pc[n] = m_backend.at(
                        {static_cast<typename decltype(m_backend
                         )::parent_t::contravariant_input_t::scalar_t>(
                             i + ((n & 4) ? 1 : 0)
                         ),
                         static_cast<typename decltype(m_backend
                         )::parent_t::contravariant_input_t::scalar_t>(
                             j + ((n & 2) ? 1 : 0)
                         ),
                         static_cast<typename decltype(m_backend
                         )::parent_t::contravariant_input_t::scalar_t>(
                             k + ((n & 1) ? 1 : 0)
                         )}
                    );
                }

                typename covariant_output_t::vector_t rv;

                for (std::size_t q = 0; q < covariant_output_t::dimensions; ++q)
                {
                    rv[q] =
                        ra * rb * rc *
                            static_cast<input_scalar_type>(pc[0][q]) +
                        ra * rb * c * static_cast<input_scalar_type>(pc[1][q]) +
                        ra * b * rc * static_cast<input_scalar_type>(pc[2][q]) +
                        ra * b * c * static_cast<input_scalar_type>(pc[3][q]) +
                        a * rb * rc * static_cast<input_scalar_type>(pc[4][q]) +
                        a * rb * c * static_cast<input_scalar_type>(pc[5][q]) +
                        a * b * rc * static_cast<input_scalar_type>(pc[6][q]) +
                        a * b * c * static_cast<input_scalar_type>(pc[7][q]);
                }

                return rv;
            } else {
                typename contravariant_output_t::vector_t is;

                for (std::size_t n = 0; n < contravariant_output_t::dimensions;
                     ++n)
                {
                    is[n] =
                        static_cast<contravariant_output_t::scalar_t>(coord[n]);
                }

                input_scalar_type vs[contravariant_output_t::dimensions];

                for (std::size_t n = 0; n < contravariant_output_t::dimensions;
                     ++n)
                {
                    vs[n] = coord[n] - std::trunc(coord[n]);
                }

                input_scalar_type rs[contravariant_output_t::dimensions];

                for (std::size_t n = 0; n < contravariant_output_t::dimensions;
                     ++n)
                {
                    rs[n] = static_cast<input_scalar_type>(1.) - vs[n];
                }

                std::remove_reference_t<typename covariant_output_t::vector_t>
                    pc[std::size_t(1) << covariant_output_t::dimensions];

                for (std::size_t n = 0;
                     n < std::size_t(1) << covariant_output_t::dimensions;
                     ++n)
                {
                    pc[n] = m_backend.at(_backend_index_helper(
                        is,
                        n,
                        std::make_index_sequence<
                            contravariant_input_t::dimensions>{}
                    ));
                }

                typename covariant_output_t::vector_t rv;

                for (std::size_t q = 0; q < covariant_output_t::dimensions; ++q)
                {
                    rv[q] = 0.f;

                    for (std::size_t n = 0;
                         n < std::size_t(1) << covariant_output_t::dimensions;
                         ++n)
                    {
                        input_scalar_type f{1.};

                        for (std::size_t m = 0;
                             m < covariant_output_t::dimensions;
                             ++m)
                        {
                            if (n & (std::size_t(1) << m)) {
                                f *= vs[m];
                            } else {
                                f *= rs[m];
                            }
                        }

                        rv[q] += f * static_cast<input_scalar_type>(pc[n][q]);
                    }
                }

                return rv;
            }
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
