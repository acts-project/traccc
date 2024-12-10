/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <iostream>
#include <memory>
#include <numeric>
#include <type_traits>

#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/concepts.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/utility/binary_io.hpp>
#include <covfie/core/utility/nd_map.hpp>
#include <covfie/core/utility/nd_size.hpp>
#include <covfie/core/vector.hpp>

namespace covfie::backend {
template <
    concepts::vector_descriptor _input_vector_t,
    concepts::field_backend _storage_t>
struct strided {
    using this_t = strided<_input_vector_t, _storage_t>;
    static constexpr bool is_initial = false;

    using backend_t = _storage_t;

    using contravariant_input_t =
        covfie::vector::array_vector_d<_input_vector_t>;
    using contravariant_output_t = typename backend_t::contravariant_input_t;
    using covariant_input_t = typename backend_t::covariant_output_t;
    using covariant_output_t = covariant_input_t;

    using coordinate_t = typename contravariant_input_t::vector_t;
    using array_t = backend_t;

    using configuration_t = utility::nd_size<contravariant_input_t::dimensions>;

    static constexpr uint32_t IO_MAGIC_HEADER = 0xAB020010;

    template <typename T>
    static std::unique_ptr<
        std::decay_t<typename backend_t::covariant_output_t::vector_t>[]>
    make_strided_copy(const T & other)
    {
        configuration_t sizes = other.get_configuration();
        std::unique_ptr<
            std::decay_t<typename backend_t::covariant_output_t::vector_t>[]>
            res = std::make_unique<std::decay_t<
                typename backend_t::covariant_output_t::vector_t>[]>(
                std::accumulate(
                    std::begin(sizes),
                    std::end(sizes),
                    1ul,
                    std::multiplies<std::size_t>()
                )
            );
        typename T::parent_t::non_owning_data_t nother(other);

        utility::nd_map<decltype(sizes)>(
            [&sizes, &nother, &res](decltype(sizes) t) {
                typename contravariant_input_t::scalar_t idx = 0;

                for (std::size_t k = 0; k < contravariant_input_t::dimensions;
                     ++k) {
                    typename contravariant_input_t::scalar_t tmp = t[k];

                    for (std::size_t l = k + 1;
                         l < contravariant_input_t::dimensions;
                         ++l)
                    {
                        tmp *= sizes[l];
                    }

                    idx += tmp;
                }

                for (std::size_t i = 0; i < covariant_output_t::dimensions; ++i)
                {
                    res[idx][i] = nother.at(t)[i];
                }
            },
            sizes
        );

        return res;
    }

    struct owning_data_t {
        using parent_t = this_t;

        owning_data_t()
            : owning_data_t(configuration_t{})
        {
        }

        owning_data_t(const owning_data_t &) = default;
        owning_data_t(owning_data_t &&) = default;
        owning_data_t & operator=(const owning_data_t &) = default;
        owning_data_t & operator=(owning_data_t &&) = default;

        template <
            typename T,
            typename B = backend_t,
            std::enable_if_t<
                std::is_same_v<
                    typename T::parent_t::configuration_t,
                    configuration_t>,
                bool> = true,
            std::enable_if_t<
                std::is_constructible_v<
                    typename B::owning_data_t,
                    std::size_t,
                    std::add_rvalue_reference_t<std::unique_ptr<std::decay_t<
                        typename B::covariant_output_t::vector_t>[]>>>,
                bool> = true>
        explicit owning_data_t(const T & o)
            : m_sizes(o.get_configuration())
            , m_storage(
                  std::accumulate(
                      std::begin(m_sizes),
                      std::end(m_sizes),
                      1ul,
                      std::multiplies<std::size_t>()
                  ),
                  make_strided_copy(o)
              )
        {
        }

        template <
            typename B = backend_t,
            std::enable_if_t<
                std::is_constructible_v<typename B::owning_data_t, std::size_t>,
                bool> = true>
        explicit owning_data_t(configuration_t conf)
            : m_sizes(conf)
            , m_storage(std::accumulate(
                  std::begin(m_sizes),
                  std::end(m_sizes),
                  static_cast<std::size_t>(1),
                  std::multiplies<std::size_t>()
              ))
        {
        }

        template <
            typename... Args,
            std::enable_if_t<(sizeof...(Args) > 0), bool> = true>
        explicit owning_data_t(parameter_pack<configuration_t, Args...> && args)
            : m_sizes(args.x)
            , m_storage(std::move(args.xs))
        {
        }

        template <
            typename T,
            std::enable_if_t<std::is_constructible_v<owning_data_t, T>, bool> =
                true>
        explicit owning_data_t(parameter_pack<T> && args)
            : owning_data_t(args.x)
        {
        }

        explicit owning_data_t(
            const configuration_t & c, typename backend_t::owning_data_t && b
        )
            : m_sizes(c)
            , m_storage(std::forward<typename backend_t::owning_data_t>(b))
        {
        }

        typename backend_t::owning_data_t & get_backend(void)
        {
            return m_storage;
        }

        const typename backend_t::owning_data_t & get_backend(void) const
        {
            return m_storage;
        }

        configuration_t get_configuration(void) const
        {
            return m_sizes;
        }

        static owning_data_t read_binary(std::istream & fs)
        {
            utility::read_io_header(fs, IO_MAGIC_HEADER);

            auto sizes = utility::read_binary<decltype(m_sizes)>(fs);
            auto be = backend_t::owning_data_t::read_binary(fs);

            utility::read_io_footer(fs, IO_MAGIC_HEADER);

            return owning_data_t(sizes, std::move(be));
        }

        static void write_binary(std::ostream & fs, const owning_data_t & o)
        {
            utility::write_io_header(fs, IO_MAGIC_HEADER);

            fs.write(
                reinterpret_cast<const char *>(&o.m_sizes),
                sizeof(decltype(o.m_sizes))
            );

            backend_t::owning_data_t::write_binary(fs, o.m_storage);

            utility::write_io_footer(fs, IO_MAGIC_HEADER);
        }

        configuration_t m_sizes;
        typename backend_t::owning_data_t m_storage;
    };

    struct non_owning_data_t {
        using parent_t = this_t;

        non_owning_data_t(const owning_data_t & o)
            : m_sizes(o.m_sizes)
            , m_storage(o.m_storage)
        {
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t at(coordinate_t c
        ) const
        {
            typename contravariant_input_t::scalar_t idx = 0;

#ifndef NDEBUG
            for (std::size_t i = 0; i < contravariant_input_t::dimensions; ++i)
            {
                assert(c[i] < m_sizes[i]);
            }
#endif

            for (std::size_t k = 0; k < contravariant_input_t::dimensions; ++k)
            {
                typename contravariant_input_t::scalar_t tmp = c[k];

                for (std::size_t l = k + 1;
                     l < contravariant_input_t::dimensions;
                     ++l)
                {
                    tmp *=
                        static_cast<typename contravariant_input_t::scalar_t>(
                            m_sizes[l]
                        );
                }

                idx += tmp;
            }

            return m_storage.at({idx});
        }

        typename backend_t::non_owning_data_t & get_backend(void)
        {
            return m_storage;
        }

        const typename backend_t::non_owning_data_t & get_backend(void) const
        {
            return m_storage;
        }

        configuration_t m_sizes;
        typename backend_t::non_owning_data_t m_storage;
    };
};
}
