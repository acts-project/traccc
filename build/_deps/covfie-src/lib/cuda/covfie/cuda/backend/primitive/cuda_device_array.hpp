/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <memory>

#include <cuda_runtime.h>

#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/concepts.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/vector.hpp>
#include <covfie/cuda/error_check.hpp>
#include <covfie/cuda/utility/memory.hpp>
#include <covfie/cuda/utility/unique_ptr.hpp>

namespace covfie::backend {
template <
    concepts::vector_descriptor _output_vector_t,
    typename _index_t = std::size_t>
struct cuda_device_array {
    using this_t = cuda_device_array<_output_vector_t, _index_t>;

    static constexpr bool is_initial = true;

    using contravariant_input_t =
        covfie::vector::scalar_d<covfie::vector::vector_d<_index_t, 1>>;
    using covariant_output_t =
        covfie::vector::array_reference_vector_d<_output_vector_t>;

    using output_vector_t = _output_vector_t;

    using value_t = typename output_vector_t::type[output_vector_t::size];
    using vector_t = std::decay_t<typename covariant_output_t::vector_t>;

    using configuration_t = utility::nd_size<1>;

    static constexpr uint32_t IO_MAGIC_HEADER = 0xAB110000;

    struct owning_data_t {
        using parent_t = this_t;

        owning_data_t()
            : m_size(0)
            , m_ptr({})
        {
        }

        owning_data_t(owning_data_t &&) = default;
        owning_data_t & operator=(owning_data_t &&) = default;

        owning_data_t & operator=(const owning_data_t & o)
        {
            m_size = o.m_size;
            m_ptr = utility::cuda::device_copy_d2d(o.m_ptr, m_size);
        }

        owning_data_t(const owning_data_t & o)
            : m_size(o.m_size)
            , m_ptr(utility::cuda::device_copy_d2d(o.m_ptr, m_size))
        {
            assert(m_size == 0 || m_ptr);

            if (o.m_ptr && m_size > 0) {
                std::memcpy(
                    m_ptr.get(), o.m_ptr.get(), m_size * sizeof(vector_t)
                );
            }
        }

        explicit owning_data_t(parameter_pack<owning_data_t> && args)
            : owning_data_t(std::move(args.x))
        {
        }

        explicit owning_data_t(parameter_pack<configuration_t> && args)
            : m_size(args.x[0])
            , m_ptr(utility::cuda::device_allocate<vector_t[]>(m_size))
        {
        }

        explicit owning_data_t(
            std::size_t size, std::unique_ptr<vector_t[]> && ptr
        )
            : m_size(size)
            , m_ptr(utility::cuda::device_copy_h2d(ptr.get(), size))
        {
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
                    if (float_width == 4) {
                        ptr[i][j] = utility::read_binary<float>(fs);
                    } else if (float_width == 8) {
                        ptr[i][j] = utility::read_binary<double>(fs);
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

        std::size_t m_size;
        utility::cuda::unique_device_ptr<vector_t[]> m_ptr;
    };

    struct non_owning_data_t {
        using parent_t = this_t;

        non_owning_data_t(const owning_data_t & o)
            : m_ptr(o.m_ptr.get())
        {
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t i) const
        {
            return m_ptr[i];
        }

        vector_t * m_ptr;
    };
};
}
