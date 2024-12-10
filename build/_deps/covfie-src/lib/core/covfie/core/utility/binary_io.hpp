/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cassert>
#include <iostream>
#include <memory>
#include <sstream>

enum class fp_type {
    IEEE754_SINGLE = 1,
    IEEE754_DOUBLE = 2
};

template <typename T>
class fp_type_id
{
};

template <>
class fp_type_id<float>
{
    static constexpr fp_type value = fp_type::IEEE754_SINGLE;
};

template <>
class fp_type_id<double>
{
    static constexpr fp_type value = fp_type::IEEE754_DOUBLE;
};

namespace covfie::utility {
static constexpr uint32_t MAGIC_HEADER = 0xC04F1EAB;
static constexpr uint32_t MAGIC_FOOTER = 0xC04F1E70;

template <typename T>
T read_binary(std::istream & fs)
{
    static_assert(
        std::is_standard_layout_v<T>, "Binary IO type must be standard layout!"
    );

    assert(fs.good() && !fs.eof() && !fs.fail() && !fs.bad());

    T rv;

    fs.read(reinterpret_cast<char *>(&rv), sizeof(T));

    return rv;
}

inline std::ostream & write_io_header(std::ostream & fs, uint32_t hdr)
{
    fs.write(
        reinterpret_cast<const char *>(&MAGIC_HEADER),
        sizeof(decltype(MAGIC_HEADER))
    );
    fs.write(reinterpret_cast<const char *>(&hdr), sizeof(decltype(hdr)));

    return fs;
}

inline std::istream & read_io_header(std::istream & fs, uint32_t hdr)
{
    uint32_t hdr1, hdr2;

    hdr1 = read_binary<uint32_t>(fs);
    hdr2 = read_binary<uint32_t>(fs);

    if (hdr1 != MAGIC_HEADER) {
        std::stringstream err;
        err << "Deserialization of covfie vector field due to non-matching "
               "global header (should be 0x"
            << std::hex << std::uppercase << MAGIC_HEADER << ", but was 0x"
            << hdr1 << ")";
        throw std::runtime_error(err.str());
    }

    if (hdr2 != hdr) {
        std::stringstream err;
        err << "Deserialization of covfie vector field due to non-matching "
               "backend header (should be 0x"
            << std::hex << std::uppercase << hdr << ", but was 0x" << hdr1
            << ")";
        throw std::runtime_error(err.str());
    }

    return fs;
}

inline std::ostream & write_io_footer(std::ostream & fs, uint32_t ftr)
{
    ftr += 0x20000000;
    fs.write(
        reinterpret_cast<const char *>(&MAGIC_FOOTER),
        sizeof(decltype(MAGIC_FOOTER))
    );
    fs.write(reinterpret_cast<const char *>(&ftr), sizeof(decltype(ftr)));

    return fs;
}

inline std::istream & read_io_footer(std::istream & fs, uint32_t ftr)
{
    ftr += 0x20000000;
    uint32_t ftr1 = 0, ftr2 = 0;

    ftr1 = read_binary<uint32_t>(fs);
    ftr2 = read_binary<uint32_t>(fs);

    if (ftr1 != MAGIC_FOOTER) {
        std::stringstream err;
        err << "Deserialization of covfie vector field due to non-matching "
               "global footer (should be 0x"
            << std::hex << std::uppercase << MAGIC_FOOTER << ", but was 0x"
            << ftr1 << ")";
        throw std::runtime_error(err.str());
    }

    if (ftr2 != ftr) {
        std::stringstream err;
        err << "Deserialization of covfie vector field due to non-matching "
               "backend footer (should be 0x"
            << std::hex << std::uppercase << ftr << ", but was 0x" << ftr2
            << ")";
        throw std::runtime_error(err.str());
    }

    return fs;
}
}
