/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/container.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <fstream>

namespace traccc {

/// Function for binary file writing
///
/// @param out_name is the output filename which includes the path
/// @param container is the input traccc container
template <typename container_t>
void write_binary(const std::string& out_name, const container_t& container) {
    std::ofstream out_file(out_name, std::ios::out | std::ios::binary);

    const auto& headers = container.get_headers();
    const auto& items = container.get_items();

    typename container_t::header_vector::size_type headers_size =
        container.size();

    // Write size
    out_file.write(reinterpret_cast<const char*>(&headers_size),
                   sizeof(typename container_t::header_vector::size_type));

    for (const auto& i : items) {
        typename container_t::item_vector::size_type items_size = i.size();

        out_file.write(reinterpret_cast<const char*>(&items_size),
                       sizeof(typename container_t::item_vector::size_type));
    }

    // Write elements
    out_file.write(reinterpret_cast<const char*>(&headers[0]),
                   headers.size() * sizeof(typename container_t::header_type));

    for (const auto& i : items) {
        out_file.write(reinterpret_cast<const char*>(&i[0]),
                       i.size() * sizeof(typename container_t::item_type));
    }

    out_file.close();
}

/// Function for binary file writing
///
/// @param in_name is the input filename which includes the path
/// @param copy is the vecem copy helper object
/// @param resource is the vecmem memory resource
template <typename container_t>
container_t read_binary(const std::string& in_name, vecmem::copy& copy,
                        vecmem::memory_resource& resource) {
    std::ifstream in_file(in_name, std::ios::binary);

    // Read size
    typename container_t::header_vector::size_type headers_size;
    in_file.read(reinterpret_cast<char*>(&headers_size),
                 sizeof(typename container_t::header_vector::size_type));

    std::vector<typename container_t::item_vector::size_type> items_size(
        headers_size);

    in_file.read(
        reinterpret_cast<char*>(&items_size[0]),
        headers_size * sizeof(typename container_t::item_vector::size_type));

    // Read element
    container_t container(&resource);
    container_buffer<typename container_t::header_type,
                     typename container_t::item_type>
        buffer{{static_cast<unsigned int>(headers_size), resource},
               {items_size, resource}};

    copy.setup(buffer.headers);
    copy.setup(buffer.items);

    in_file.read(reinterpret_cast<char*>(buffer.headers.ptr()),
                 headers_size * sizeof(typename container_t::header_type));

    for (std::size_t i = 0; i < headers_size; i++) {
        in_file.read(
            reinterpret_cast<char*>((buffer.items.host_ptr() + i)->ptr()),
            items_size[i] * sizeof(typename container_t::item_type));
    }

    copy(buffer.headers, container.get_headers());
    copy(buffer.items, container.get_items());

    in_file.close();

    return container;
}

}  // namespace traccc