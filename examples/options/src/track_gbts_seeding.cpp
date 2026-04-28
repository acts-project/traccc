/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/track_gbts_seeding.hpp"

#include "traccc/examples/utils/printable.hpp"

// System include(s).
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

track_gbts_seeding::track_gbts_seeding() : interface("GBTS Options") {
	// get linking scheme
    m_desc.add_options()(
        "gbts_config_dir",
        boost::program_options::value(&config_dir)->default_value(config_dir),
        "directory for gbts config files");
	m_desc.add_options()("useGBTS",
		boost::program_options::bool_switch(&useGBTS),
		"use gbts algorithm");	
	// set CLI tunings for graph building
	m_desc.add_options()(
        "max_edges_factor",
        po::value(&gbts_config.max_edges_factor)->default_value(gbts_config.max_edges_factor),
        "number of edges allocated for per node ");
	
	m_desc.add_options()(
        "min_delta_phi",
        po::value(&gbts_config.graph_building_params.min_delta_phi)->default_value(gbts_config.graph_building_params.min_delta_phi),
        "min_delta_phi for the sliding window [rads]");
	m_desc.add_options()(
        "dphi_coeff for the sliding window",
        po::value(&gbts_config.graph_building_params.max_z0)->default_value(gbts_config.graph_building_params.dphi_coeff),
        "dphi_coeff for the sliding window");
	m_desc.add_options()(
        "min_delta_phi_low_dr",
        po::value(&gbts_config.graph_building_params.max_z0)->default_value(gbts_config.graph_building_params.min_delta_phi_low_dr),
        "min_delta_phi_low_dr for the sliding window [rads]");
	m_desc.add_options()(
        "dphi_coeff_low_dr",
        po::value(&gbts_config.graph_building_params.max_z0)->default_value(gbts_config.graph_building_params.dphi_coeff),
        "dphi_coeff_low_dr for the sliding window");
	m_desc.add_options()(
        "max_Kappa",
        po::value(&gbts_config.graph_building_params.max_z0)->default_value(gbts_config.graph_building_params.max_Kappa),
        "max curvature for an edge + origin triplet [1/mm]");
	
	m_desc.add_options()(
        "min_pt",
        po::value(&min_pt)->default_value(min_pt),
        "min_pt to scale other cuts by with refrence to 900 MeV [MeV]");
	
	m_desc.add_options()(
        "min_z0",
        po::value(&gbts_config.graph_building_params.min_z0)->default_value(gbts_config.graph_building_params.min_z0),
        "min projected z0 for an edge [mm]");
	m_desc.add_options()(
        "max_z0",
        po::value(&gbts_config.graph_building_params.max_z0)->default_value(gbts_config.graph_building_params.max_z0),
        "max projected z0 for an edge [mm]");
	m_desc.add_options()(
        "maxOuterRadius projected for an edge",
        po::value(&gbts_config.graph_building_params.maxOuterRadius)->default_value(gbts_config.graph_building_params.maxOuterRadius),
        "maxOuterRadius [mm]");
	
	m_desc.add_options()(
        "cut_dphi_max",
        po::value(&gbts_config.graph_building_params.max_z0)->default_value(gbts_config.graph_building_params.cut_dphi_max),
        "cut_dphi_max for edge matching [rads]");
	m_desc.add_options()(
        "cut_dcurv_max",
        po::value(&gbts_config.graph_building_params.max_z0)->default_value(gbts_config.graph_building_params.cut_dcurv_max),
        "cut_dcurv_max for edge matching [1/mm]");
	m_desc.add_options()(
        "cut_tau_ratio_max",
        po::value(&gbts_config.graph_building_params.max_z0)->default_value(gbts_config.graph_building_params.cut_tau_ratio_max),
        "cut_tau_ratio_max for edge matching");
	m_desc.add_options()(
        "max_num_neighbours",
        po::value(&gbts_config.graph_building_params.min_delta_phi)->default_value(gbts_config.max_num_neighbours),
        "max connected neighbours for each edge");
	// set CLI tuning for seed extraction kalman filter
	m_desc.add_options()(
        "sigmaMS",
        po::value(&gbts_config.seed_extraction_params.sigmaMS)->default_value(gbts_config.seed_extraction_params.sigmaMS),
        "sigmaMS for a 900MeV track at eta=0, used in seed fitting");
	m_desc.add_options()(
        "radLen",
        po::value(&gbts_config.seed_extraction_params.radLen)->default_value(gbts_config.seed_extraction_params.radLen),
        "% radLen per layer for seed fitting");
	m_desc.add_options()(
        "sigma_x",
        po::value(&gbts_config.seed_extraction_params.sigma_x)->default_value(gbts_config.seed_extraction_params.sigma_x),
        "sigma_x for a seed's fit");
	m_desc.add_options()(
        "sigma_y",
        po::value(&gbts_config.seed_extraction_params.sigma_y)->default_value(gbts_config.seed_extraction_params.sigma_y),
        "sigma_y for a seed's fit");
	m_desc.add_options()(
        "maxDChi2_x" ,
        po::value(&gbts_config.seed_extraction_params.maxDChi2_x)->default_value(gbts_config.seed_extraction_params.maxDChi2_x),
        "maxDChi2_x for a seed's fit");
	m_desc.add_options()(
        "maxDChi2_y ",
        po::value(&gbts_config.seed_extraction_params.maxDChi2_y)->default_value(gbts_config.seed_extraction_params.maxDChi2_y),
        "maxDChi2_y for a seed's fit");
	m_desc.add_options()(
        "add_hit",
        po::value(&gbts_config.seed_extraction_params.add_hit)->default_value(gbts_config.seed_extraction_params.add_hit),
        "base quality to add for each hit on seed");
	m_desc.add_options()(
        "inv_max_curvature",
        po::value(&gbts_config.seed_extraction_params.inv_max_curvature)->default_value(gbts_config.seed_extraction_params.inv_max_curvature),
        "inv_max_curvature to stop following a seed [1/mm]");
	m_desc.add_options()(
        "max_z0",
        po::value(&gbts_config.seed_extraction_params.max_z0)->default_value(gbts_config.seed_extraction_params.max_z0),
        "max_z0 to stop following a seed [mm]");

}

track_gbts_seeding::operator gbts_seedfinder_config() const {

    return gbts_config;
}

void track_gbts_seeding::read(const boost::program_options::variables_map &) {
    // fill config
    if (!useGBTS) {
        return;
    }
	// config info from file
    std::vector<std::pair<uint64_t, short>> barcodeBinning;
    std::vector<std::pair<int, std::vector<int>>> binTables;
    traccc::device::gbts_layerInfo layerInfo;
    
	std::ifstream barcodeBinningFile(
        std::filesystem::path(config_dir + "/barcodeBinning.txt"));

    unsigned int nBarcodes = 0;
    barcodeBinningFile >> nBarcodes;
    barcodeBinning.reserve(nBarcodes);

    std::pair<uint64_t, short> barcodeLayerPair;
    for (; nBarcodes > 0u; --nBarcodes) {
        barcodeBinningFile >> barcodeLayerPair.first;
        barcodeBinningFile >> barcodeLayerPair.second;

        barcodeBinning.push_back(barcodeLayerPair);
    }

    std::ifstream binTablesFile(
        std::filesystem::path(config_dir + "/binTables.txt"));

    unsigned int nBinPairs = 0;
    binTablesFile >> nBinPairs;
    binTables.reserve(nBinPairs);
    int bin1 = 0;
    std::vector<int> bin2 = {0};
    for (; nBinPairs > 0; --nBinPairs) {
        binTablesFile >> bin1;
        binTablesFile >> bin2[0];
        binTables.emplace_back(bin1, bin2);
    }

    std::ifstream layerInfoFile(
        std::filesystem::path(config_dir + "/layerInfo.txt"));

    unsigned int nLayers = 0;
    layerInfoFile >> nLayers;
    layerInfo.reserve(nLayers);
    int type = 0;
    std::array<int, 2> info = {0, 0};
    std::array<float, 2> geo = {0, 0};
    for (; nLayers > 0u; --nLayers) {
        layerInfoFile >> type;
        layerInfoFile >> info[0] >> info[1];
        layerInfoFile >> geo[0] >> geo[1];
        layerInfo.addLayer(static_cast<char>(type), info[0], info[1], geo[0],
                           geo[1]);
    }
	// Set linking scheme
	gbts_config.setLinkingScheme(binTables, layerInfo,
                                      barcodeBinning, min_pt,
                                      getDefaultLogger("GBTSconfig"));	
}

std::unique_ptr<configuration_printable> track_gbts_seeding::as_printable()
    const {
    auto cat = std::make_unique<configuration_category>(m_description);

    cat->add_child(std::make_unique<configuration_kv_pair>(
        "using gbts algorithm ", std::to_string(useGBTS)));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "gbts config directory ", config_dir));

    return cat;
}

}  // namespace traccc::opts
