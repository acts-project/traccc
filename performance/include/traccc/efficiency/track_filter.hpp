/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <string>
#include <traccc/definitions/primitives.hpp>
#include <traccc/edm/particle.hpp>

namespace traccc {
struct track_filter {
    virtual std::string get_name() const = 0;
    virtual bool operator()(const particle &) const = 0;
};

struct simple_charged_eta_pt_cut : track_filter {
    simple_charged_eta_pt_cut(scalar, scalar);

    virtual std::string get_name() const override final;

    virtual bool operator()(const particle &) const override final;

    private:
    scalar m_eta, m_pT;
};
}  // namespace traccc
