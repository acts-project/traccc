/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cstddef>
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/data/vector_view.hpp>

#include "traccc/definitions/qualifiers.hpp"

namespace traccc::device {
/*
 * This file contains classes that can be used to access memory in a strided
 * way with arbitrary, standard C++ structs. The design is similar to that of
 * vecmem vectors, where memory is owned by a buffer object which can be
 * passed to a kernel as a view object. The kernel can then create a device
 * vector out of it in order to load and store elements.
 *
 * The purpose of this class is to provide optimal coalescing even when using
 * large structs. Operates in a similar fashion to CUDA local memory.
 */

template <typename T, std::size_t MaxStride>
class strided_vector_view;

/**
 * @brief Buffer object for a strided vector.
 *
 * Simply wraps a vecmem buffer, and must be created from a vecmem buffer.
 */
template <typename T, std::size_t MaxStride = 4>
class strided_vector_buffer {
    static_assert(MaxStride == 8 || MaxStride == 4 || MaxStride == 2 ||
                  MaxStride == 1);

    public:
    TRACCC_HOST_DEVICE strided_vector_buffer(vecmem::data::vector_buffer<T> &&b)
        : m_buffer(std::move(b)) {}

    private:
    friend strided_vector_view<T, MaxStride>;
    vecmem::data::vector_buffer<T> m_buffer;
};

template <typename T, std::size_t MaxStride>
class strided_vector_device;

/**
 * @brief View object for a strided vector.
 *
 * Simply wraps a vecmem view. Implicitly creatable from a strided vector
 * buffer in the same way a vecmem buffer can be turned into a vecmem view.
 */
template <typename T, std::size_t MaxStride = 4>
class strided_vector_view {
    static_assert(MaxStride == 8 || MaxStride == 4 || MaxStride == 2 ||
                  MaxStride == 1);

    public:
    TRACCC_HOST_DEVICE strided_vector_view(
        const strided_vector_buffer<T, MaxStride> &b)
        : m_view(b.m_buffer) {}

    private:
    friend strided_vector_device<T, MaxStride>;

    vecmem::data::vector_view<T> m_view;
};

/**
 * @brief Device-side accessor for a strided vector.
 *
 * This class supports loading and storing large structs stored in memory in
 * a strided way. When accessing data in such a vector, we always load at most
 * `MaxStride` bytes of adjacent memory.
 */
template <typename T, std::size_t MaxStride = 4>
class strided_vector_device {
    static_assert(MaxStride == 8 || MaxStride == 4 || MaxStride == 2 ||
                  MaxStride == 1);

    using SIZE_8_TYPE = long;
    using SIZE_4_TYPE = int;
    using SIZE_2_TYPE = short;
    using SIZE_1_TYPE = char;

    static_assert(sizeof(SIZE_8_TYPE) == 8);
    static_assert(sizeof(SIZE_4_TYPE) == 4);
    static_assert(sizeof(SIZE_2_TYPE) == 2);
    static_assert(sizeof(SIZE_1_TYPE) == 1);

    public:
    TRACCC_HOST_DEVICE strided_vector_device(
        strided_vector_view<T, MaxStride> &v)
        : m_ptr(v.m_view.ptr()), m_size(v.m_view.size()) {}

    TRACCC_HOST_DEVICE strided_vector_device(void *ptr, unsigned int size)
        : m_ptr(ptr), m_size(size) {}

    /**
     * @brief Store the given object at the given index.
     */
    TRACCC_HOST_DEVICE void store(const T &in, unsigned int i) {
        unsigned int b0 = 0;
        constexpr unsigned int b = sizeof(T);
        char *const out_byte_ptr = reinterpret_cast<char *>(m_ptr);
        const char *const in_byte_ptr = reinterpret_cast<const char *>(&in);

        while (b0 < b) {
            unsigned int delta = b - b0;
            if (MaxStride >= 8 && delta >= 8) {
                *reinterpret_cast<SIZE_8_TYPE *>(out_byte_ptr + b0 * m_size +
                                                 i * 8) =
                    *reinterpret_cast<const SIZE_8_TYPE *>(in_byte_ptr + b0);
                b0 += 8;
            } else if (MaxStride >= 4 && delta >= 4) {
                *reinterpret_cast<SIZE_4_TYPE *>(out_byte_ptr + b0 * m_size +
                                                 i * 4) =
                    *reinterpret_cast<const SIZE_4_TYPE *>(in_byte_ptr + b0);
                b0 += 4;
            } else if (MaxStride >= 2 && delta >= 2) {
                *reinterpret_cast<SIZE_2_TYPE *>(out_byte_ptr + b0 * m_size +
                                                 i * 2) =
                    *reinterpret_cast<const SIZE_2_TYPE *>(in_byte_ptr + b0);
                b0 += 2;
            } else if (MaxStride >= 1 && delta >= 1) {
                *reinterpret_cast<SIZE_1_TYPE *>(out_byte_ptr + b0 * m_size +
                                                 i) =
                    *reinterpret_cast<const SIZE_1_TYPE *>(in_byte_ptr + b0);
                b0 += 1;
            }
        }
    }

    /**
     * @brief Load an object from the given index.
     */
    TRACCC_HOST_DEVICE T load(unsigned int i) {
        T out;
        unsigned int b0 = 0;
        constexpr unsigned int b = sizeof(T);
        char *const out_byte_ptr = reinterpret_cast<char *>(&out);
        const char *const in_byte_ptr = reinterpret_cast<const char *>(m_ptr);

        while (b0 < b) {
            unsigned int delta = b - b0;
            if (MaxStride >= 8 && delta >= 8) {
                *reinterpret_cast<SIZE_8_TYPE *>(out_byte_ptr + b0) =
                    *reinterpret_cast<const SIZE_8_TYPE *>(in_byte_ptr +
                                                           b0 * m_size + i * 8);
                b0 += 8;
            } else if (MaxStride >= 4 && delta >= 4) {
                *reinterpret_cast<SIZE_4_TYPE *>(out_byte_ptr + b0) =
                    *reinterpret_cast<const SIZE_4_TYPE *>(in_byte_ptr +
                                                           b0 * m_size + i * 4);
                b0 += 4;
            } else if (MaxStride >= 2 && delta >= 2) {
                *reinterpret_cast<SIZE_2_TYPE *>(out_byte_ptr + b0) =
                    *reinterpret_cast<const SIZE_2_TYPE *>(in_byte_ptr +
                                                           b0 * m_size + i * 2);
                b0 += 2;
            } else if (MaxStride >= 1 && delta >= 1) {
                *reinterpret_cast<SIZE_1_TYPE *>(out_byte_ptr + b0) =
                    *reinterpret_cast<const SIZE_1_TYPE *>(in_byte_ptr +
                                                           b0 * m_size + i);
                b0 += 1;
            }
        }
        return out;
    }

    private:
    void *m_ptr;
    unsigned int m_size;
};
}  // namespace traccc::device
