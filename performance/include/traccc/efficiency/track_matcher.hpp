/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <optional>
#include <string>
#include <traccc/definitions/primitives.hpp>
#include <traccc/edm/particle.hpp>
#include <vector>

namespace traccc {
struct track_matcher {
    virtual std::string get_name() const = 0;
    virtual std::optional<uint64_t> operator()(
        const std::vector<std::vector<uint64_t>>&) const = 0;
};

struct stepped_percentage : track_matcher {
    stepped_percentage(scalar);

    virtual std::string get_name() const override final;

    virtual std::optional<uint64_t> operator()(
        const std::vector<std::vector<uint64_t>>&) const override final;

    private:
    scalar m_min_ratio;
};

struct exact : track_matcher {
    exact();

    virtual std::string get_name() const override final;

    virtual std::optional<uint64_t> operator()(
        const std::vector<std::vector<uint64_t>>&) const override final;
};
}  // namespace traccc
