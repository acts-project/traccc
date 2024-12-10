/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <algorithm>
#include <climits>
#include <iostream>
#include <memory>
#include <numeric>
#include <type_traits>

#include <covfie/core/array.hpp>
#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/concepts.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/utility/binary_io.hpp>
#include <covfie/core/utility/nd_map.hpp>
#include <covfie/core/utility/nd_size.hpp>
#include <covfie/core/utility/numeric.hpp>
#include <covfie/core/vector.hpp>

namespace covfie::backend {
template <
    concepts::vector_descriptor _input_vector_t,
    concepts::field_backend _storage_t>
struct hilbert {
    using this_t = hilbert<_input_vector_t, _storage_t>;
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

    static constexpr uint32_t IO_MAGIC_HEADER = 0xAB020004;

    static_assert(
        contravariant_input_t::dimensions == 2,
        "Number of dimensions for input must be exactly two."
    );

    COVFIE_DEVICE static void
    rot(std::size_t n,
        std::size_t * x,
        std::size_t * y,
        std::size_t rx,
        std::size_t ry)
    {
        if (ry == 0) {
            if (rx == 1) {
                *x = n - 1 - *x;
                *y = n - 1 - *y;
            }

            std::size_t t = *x;
            *x = *y;
            *y = t;
        }
    }

    COVFIE_DEVICE static std::size_t calculate_index(
        coordinate_t c,
        utility::nd_size<contravariant_input_t::dimensions> sizes
    )
    {
        // Borrowed from https://en.wikipedia.org/wiki/Hilbert_curve

        std::size_t rx, ry, s, d = 0;

        std::size_t x = c[0];
        std::size_t y = c[1];

        // TODO: `sizes[0]` has to equal `sizes[1]`.
        for (s = sizes[0] / 2; s > 0; s /= 2) {
            rx = (x & s) > 0;
            ry = (y & s) > 0;
            d += s * s * ((3 * rx) ^ ry);
            rot(sizes[0], &x, &y, rx, ry);
        }

        return d;
    }

    template <typename T>
    static std::unique_ptr<
        std::decay_t<typename backend_t::covariant_output_t::vector_t>[]>
    make_hilbert_copy(const T & other)
    {
        configuration_t sizes = other.get_configuration();

        std::unique_ptr<
            std::decay_t<typename backend_t::covariant_output_t::vector_t>[]>
            res = std::make_unique<std::decay_t<
                typename backend_t::covariant_output_t::vector_t>[]>(
                utility::ipow(
                    utility::round_pow2(
                        *std::max_element(sizes.begin(), sizes.end())
                    ),
                    contravariant_input_t::dimensions
                )
            );
        typename T::parent_t::non_owning_data_t nother(other);

        utility::nd_map<decltype(sizes)>(
            [&nother, &res](decltype(sizes) t) {
                coordinate_t c;

                for (std::size_t i = 0; i < contravariant_input_t::dimensions;
                     ++i) {
                    c[i] = t[i];
                }

                std::size_t idx = calculate_index(c);

                for (std::size_t i = 0; i < covariant_output_t::dimensions; ++i)
                {
                    res[idx][i] = nother.at(c)[i];
                }
            },
            sizes
        );

        return res;
    }

    struct owning_data_t {
        using parent_t = this_t;

        owning_data_t() = default;
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
                  utility::ipow(
                      utility::round_pow2(
                          *std::max_element(m_sizes.begin(), m_sizes.end())
                      ),
                      contravariant_input_t::dimensions
                  ),
                  make_hilbert_copy(o)
              )
        {
        }

        template <
            typename... Args,
            typename B = backend_t,
            std::enable_if_t<
                !std::
                    is_constructible_v<typename B::owning_data_t, std::size_t>,
                bool> = true>
        explicit owning_data_t(configuration_t conf, Args... args)
            : m_sizes(conf)
            , m_storage(std::forward<Args>(args)...)
        {
        }

        template <typename... Args>
        explicit owning_data_t(parameter_pack<configuration_t, Args...> && args)
            : m_sizes(args.x)
            , m_storage(std::move(args.xs))
        {
        }

        explicit owning_data_t(parameter_pack<owning_data_t> && conf)
            : owning_data_t(std::move(conf.x))
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
            auto be = decltype(m_storage)::read_binary(fs);

            utility::read_io_footer(fs, IO_MAGIC_HEADER);

            return owning_data_t(sizes, std::move(be));
        }

        static void write_binary(std::ostream & fs, const owning_data_t & o)
        {
            utility::write_io_header(fs, IO_MAGIC_HEADER);

            fs.write(
                reinterpret_cast<const char *>(&o.m_sizes),
                sizeof(decltype(m_sizes))
            );

            decltype(m_storage)::write_binary(fs, o.m_storage);

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
#ifndef NDEBUG
            for (std::size_t i = 0; i < contravariant_input_t::dimensions; ++i)
            {
                assert(c[i] < m_sizes[i]);
            }
#endif

            return m_storage.at(calculate_index(c));
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
