/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cassert>
#include <cstring>
#include <memory>
#include <utility>

#include <covfie/core/concepts.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/utility/binary_io.hpp>
#include <covfie/core/utility/nd_size.hpp>
#include <covfie/core/vector.hpp>

namespace covfie::backend {
template <
    concepts::vector_descriptor _output_vector_t,
    typename _index_t = std::size_t>
struct array {
    using this_t = array<_output_vector_t, _index_t>;
    static constexpr bool is_initial = true;

    using contravariant_input_t =
        covfie::vector::scalar_d<covfie::vector::vector_d<_index_t, 1>>;
    using covariant_output_t =
        covfie::vector::array_reference_vector_d<_output_vector_t>;

    using vector_t = std::decay_t<typename covariant_output_t::vector_t>;

    using configuration_t = utility::nd_size<contravariant_input_t::dimensions>;

    static constexpr uint32_t IO_MAGIC_HEADER = 0xAB010000;

    struct owning_data_t {
        using parent_t = this_t;

        owning_data_t()
            : m_size(0)
            , m_ptr({})
        {
        }

        owning_data_t(owning_data_t &&) = default;
        owning_data_t & operator=(owning_data_t &&) = default;

        explicit owning_data_t(std::size_t n)
            : m_size(n)
            , m_ptr(std::make_unique<vector_t[]>(n))
        {
        }

        explicit owning_data_t(configuration_t conf)
            : owning_data_t(conf[0])
        {
        }

        explicit owning_data_t(parameter_pack<owning_data_t> && conf)
            : owning_data_t(std::move(conf.x))
        {
        }

        explicit owning_data_t(parameter_pack<configuration_t> && conf)
            : owning_data_t(conf.x[0])
        {
        }

        explicit owning_data_t(
            std::size_t size, std::unique_ptr<vector_t[]> && ptr
        )
            : m_size(size)
            , m_ptr(std::move(ptr))
        {
        }

        owning_data_t(const owning_data_t & o)
            : m_size(o.m_size)
            , m_ptr(std::make_unique<vector_t[]>(m_size))
        {
            assert(m_size == 0 || m_ptr);

            if (o.m_ptr && m_size > 0) {
                std::memcpy(
                    m_ptr.get(), o.m_ptr.get(), m_size * sizeof(vector_t)
                );
            }
        }

        owning_data_t & operator=(const owning_data_t & o)
        {
            m_size = o.m_size;
            m_ptr = std::make_unique<vector_t[]>(m_size);

            assert(m_size == 0 || m_ptr);

            if (o.m_ptr && m_size > 0) {
                std::memcpy(
                    m_ptr.get(), o.m_ptr.get(), m_size * sizeof(vector_t)
                );
            }
        }

        configuration_t get_configuration() const
        {
            return {m_size};
        }

        static owning_data_t read_binary(std::istream & fs)
        {
            utility::read_io_header(fs, IO_MAGIC_HEADER);

            uint32_t float_width = utility::read_binary<uint32_t>(fs);

            if (float_width != 4 && float_width != 8) {
                throw std::runtime_error(
                    "Float type is neither IEEE 754 single- nor "
                    "double-precision, binary input is not supported."
                );
            }

            auto size =
                utility::read_binary<std::decay_t<decltype(m_size)>>(fs);
            std::unique_ptr<vector_t[]> ptr =
                std::make_unique<vector_t[]>(size);

            for (std::size_t i = 0; i < size; ++i) {
                for (std::size_t j = 0; j < _output_vector_t::size; ++j) {
                    using scalar_t = typename _output_vector_t::type;
                    if (float_width == 4) {
                        ptr[i][j] =
                            static_cast<scalar_t>(utility::read_binary<float>(fs
                            ));
                    } else if (float_width == 8) {
                        ptr[i][j] = static_cast<scalar_t>(
                            utility::read_binary<double>(fs)
                        );
                    } else {
                        throw std::logic_error("Float width is unexpected.");
                    }
                }
            }

            utility::read_io_footer(fs, IO_MAGIC_HEADER);

            return owning_data_t(size, std::move(ptr));
        }

        static void write_binary(std::ostream & fs, const owning_data_t & o)
        {
            utility::write_io_header(fs, IO_MAGIC_HEADER);

            uint32_t float_width;

            if constexpr (std::
                              is_same_v<typename _output_vector_t::type, float>)
            {
                float_width = 4;
            } else if constexpr (std::is_same_v<
                                     typename _output_vector_t::type,
                                     double>)
            {
                float_width = 8;
            } else {
                throw std::logic_error(
                    "Float type is neither IEEE 754 single- nor "
                    "double-precision, binary output is not supported."
                );
            }

            fs.write(
                reinterpret_cast<const char *>(&float_width),
                sizeof(std::decay_t<decltype(float_width)>)
            );

            fs.write(
                reinterpret_cast<const char *>(&o.m_size),
                sizeof(std::decay_t<decltype(o.m_size)>)
            );

            for (std::size_t i = 0; i < o.m_size; ++i) {
                for (std::size_t j = 0; j < _output_vector_t::size; ++j) {
                    fs.write(
                        reinterpret_cast<const char *>(&o.m_ptr[i][j]),
                        sizeof(typename _output_vector_t::type)
                    );
                }
            }

            utility::write_io_footer(fs, IO_MAGIC_HEADER);
        }

        uint64_t m_size;
        std::unique_ptr<vector_t[]> m_ptr;
    };

    struct non_owning_data_t {
        using parent_t = this_t;

        non_owning_data_t(const owning_data_t & o)
            : m_size(o.m_size)
            , m_ptr(o.m_ptr.get())
        {
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t i) const
        {
            assert(i < m_size);
            return m_ptr[i];
        }

        uint64_t m_size;
        typename decltype(owning_data_t::m_ptr)::pointer m_ptr;
    };
};
}
