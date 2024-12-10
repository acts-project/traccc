/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Detray test include(s)
#include "detray/options/options_handling.hpp"
#include "detray/test/utils/simulation/event_generator/random_track_generator_config.hpp"
#include "detray/test/utils/simulation/event_generator/uniform_track_generator_config.hpp"

// Boost
#include "detray/options/boost_program_options.hpp"

// System include(s)
#include <stdexcept>
#include <vector>

namespace detray::options {

/// Add options for detray event generation
template <>
void add_options<uniform_track_generator_config>(
    boost::program_options::options_description &desc,
    const uniform_track_generator_config &cfg) {

    desc.add_options()(
        "phi_steps",
        boost::program_options::value<std::size_t>()->default_value(
            cfg.phi_steps()),
        "No. phi steps for particle gun")(
        "eta_steps",
        boost::program_options::value<std::size_t>()->default_value(
            cfg.eta_steps()),
        "No. eta steps for particle gun")(
        "eta_range",
        boost::program_options::value<std::vector<float>>()->multitoken(),
        "Min, Max range of eta values for particle gun")(
        "randomize_charge", "Randomly flip charge sign per track")(
        "origin",
        boost::program_options::value<std::vector<float>>()->multitoken(),
        "Coordintates for particle gun origin position [mm]")(
        "p_tot",
        boost::program_options::value<float>()->default_value(
            static_cast<float>(cfg.m_p_mag) / unit<float>::GeV),
        "Total momentum of the test particle [GeV]")(
        "p_T",
        boost::program_options::value<float>()->default_value(
            static_cast<float>(cfg.m_p_mag) / unit<float>::GeV),
        "Transverse momentum of the test particle [GeV]");
}

/// Add options for detray event generation
template <>
void configure_options<uniform_track_generator_config>(
    boost::program_options::variables_map &vm,
    uniform_track_generator_config &cfg) {

    cfg.phi_steps(vm["phi_steps"].as<std::size_t>());
    cfg.eta_steps(vm["eta_steps"].as<std::size_t>());
    cfg.randomize_charge(vm.count("randomize_charge"));

    if (vm.count("eta_range")) {
        const auto eta_range = vm["eta_range"].as<std::vector<float>>();
        if (eta_range.size() == 2u) {
            cfg.eta_range(eta_range[0], eta_range[1]);
        } else {
            throw std::invalid_argument("Eta range needs two arguments");
        }
    }
    if (vm.count("origin")) {
        const auto origin = vm["origin"].as<std::vector<float>>();
        if (origin.size() == 3u) {
            cfg.origin({origin[0] * unit<float>::mm,
                        origin[1] * unit<float>::mm,
                        origin[2] * unit<float>::mm});
        } else {
            throw std::invalid_argument(
                "Particle gun origin needs three arguments");
        }
    }
    if (!vm["p_T"].defaulted() && !vm["p_tot"].defaulted()) {
        throw std::invalid_argument(
            "Transverse and total momentum cannot be specified at the same "
            "time");
    }
    if (!vm["p_T"].defaulted()) {
        cfg.p_T(vm["p_T"].as<float>() * unit<float>::GeV);
    } else {
        cfg.p_tot(vm["p_tot"].as<float>() * unit<float>::GeV);
    }
}

/// Add options for detray event generation
template <>
void add_options<random_track_generator_config>(
    boost::program_options::options_description &desc,
    const random_track_generator_config &cfg) {

    desc.add_options()(
        "n_tracks",
        boost::program_options::value<std::size_t>()->default_value(
            cfg.n_tracks()),
        "No. of tracks for particle gun")(
        "theta_range",
        boost::program_options::value<std::vector<float>>()->multitoken(),
        "Min, Max range of theta values for particle gun")(
        "eta_range",
        boost::program_options::value<std::vector<float>>()->multitoken(),
        "Min, Max range of eta values for particle gun")(
        "randomize_charge", "Randomly flip charge sign per track")(
        "origin",
        boost::program_options::value<std::vector<float>>()->multitoken(),
        "Coordintates for particle gun origin position")(
        "p_tot",
        boost::program_options::value<float>()->default_value(
            static_cast<float>(cfg.mom_range()[0]) / unit<float>::GeV),
        "Total momentum of the test particle [GeV]")(
        "p_T",
        boost::program_options::value<float>()->default_value(
            static_cast<float>(cfg.mom_range()[0]) / unit<float>::GeV),
        "Transverse momentum of the test particle [GeV]");
}

/// Add options for detray event generation
template <>
void configure_options<random_track_generator_config>(
    boost::program_options::variables_map &vm,
    random_track_generator_config &cfg) {

    cfg.n_tracks(vm["n_tracks"].as<std::size_t>());
    cfg.randomize_charge(vm.count("randomize_charge"));

    if (vm.count("eta_range") && vm.count("theta_range")) {
        throw std::invalid_argument(
            "Eta range and theta range cannot be specified at the same time");
    } else if (vm.count("eta_range")) {
        const auto eta_range = vm["eta_range"].as<std::vector<float>>();
        if (eta_range.size() == 2u) {
            float min_theta{2.f * std::atan(std::exp(-eta_range[0]))};
            float max_theta{2.f * std::atan(std::exp(-eta_range[1]))};

            // Wrap around
            if (min_theta > max_theta) {
                float tmp{min_theta};
                min_theta = max_theta;
                max_theta = tmp;
            }

            cfg.theta_range(min_theta, max_theta);
        } else {
            throw std::invalid_argument("Eta range needs two arguments");
        }
    } else if (vm.count("theta_range")) {
        const auto theta_range = vm["theta_range"].as<std::vector<float>>();
        if (theta_range.size() == 2u) {
            cfg.theta_range(theta_range[0], theta_range[1]);
        } else {
            throw std::invalid_argument("Theta range needs two arguments");
        }
    }
    if (vm.count("origin")) {
        const auto origin = vm["origin"].as<std::vector<float>>();
        if (origin.size() == 3u) {
            cfg.origin({origin[0], origin[1], origin[2]});
        } else {
            throw std::invalid_argument(
                "Particle gun origin needs three arguments");
        }
    }
    if (!vm["p_T"].defaulted() && !vm["p_tot"].defaulted()) {
        throw std::invalid_argument(
            "Transverse and total momentum cannot be specified at the same "
            "time");
    }
    if (!vm["p_T"].defaulted()) {
        cfg.p_T(vm["p_T"].as<float>() * unit<float>::GeV);
    } else {
        cfg.p_tot(vm["p_tot"].as<float>() * unit<float>::GeV);
    }
}

}  // namespace detray::options
