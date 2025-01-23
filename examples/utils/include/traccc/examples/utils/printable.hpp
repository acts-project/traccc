/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace traccc {
class configuration_printable {
    public:
    virtual void print() const;

    virtual void print_impl(std::string self_prefix, std::string child_prefix,
                            std::size_t prefix_len,
                            std::size_t max_key_width) const = 0;

    virtual std::size_t get_max_key_width_impl() const = 0;

    virtual ~configuration_printable();
};

class configuration_category final : public configuration_printable {
    public:
    explicit configuration_category(std::string n);

    void add_child(std::unique_ptr<configuration_printable>&& elem);

    void print_impl(std::string self_prefix, std::string child_prefix,
                    std::size_t prefix_len,
                    std::size_t max_key_width) const final;

    std::size_t get_max_key_width_impl() const final;

    ~configuration_category() final;

    private:
    std::string name;
    std::vector<std::unique_ptr<configuration_printable>> elements;
};

class configuration_list final : public configuration_printable {
    public:
    configuration_list();

    void add_child(std::unique_ptr<configuration_printable>&& elem);

    void print_impl(std::string self_prefix, std::string child_prefix,
                    std::size_t prefix_len,
                    std::size_t max_key_width) const final;

    std::size_t get_max_key_width_impl() const final;

    ~configuration_list() final;

    private:
    std::vector<std::unique_ptr<configuration_printable>> elements;
};

class configuration_kv_pair final : public configuration_printable {
    public:
    configuration_kv_pair(std::string k, std::string v);

    void print_impl(std::string self_prefix, std::string,
                    std::size_t prefix_len,
                    std::size_t max_key_width) const final;

    std::size_t get_max_key_width_impl() const final;

    ~configuration_kv_pair() final;

    private:
    std::string key;
    std::string value;
};
}  // namespace traccc
