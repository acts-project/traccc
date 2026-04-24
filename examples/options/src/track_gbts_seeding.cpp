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

track_gbts_seeding::track_gbts_seeding() : interface("GBTS Options") {
    m_desc.add_options()("use-gbts",
                         boost::program_options::bool_switch(&m_use_gbts),
                         "Use gbts algorithm");

    m_desc.add_options()("gbts-config-dir",
                         boost::program_options::value(&m_config_dir)
                             ->default_value(m_config_dir),
                         "Directory for GBTS config files");
}

void track_gbts_seeding::read(const boost::program_options::variables_map &) {
    // fill config
    if (!m_use_gbts) {
        return;
    }
    std::ifstream barcodeBinningFile(
        std::filesystem::path(m_config_dir + "/barcodeBinning.txt"));

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
        std::filesystem::path(m_config_dir + "/binTables.txt"));

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
        std::filesystem::path(m_config_dir + "/layerInfo.txt"));

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
}

std::unique_ptr<configuration_printable> track_gbts_seeding::as_printable()
    const {
    auto cat = std::make_unique<configuration_category>(m_description);

    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Using GBTS algorithm", std::to_string(m_use_gbts)));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "GBTS config directory ", m_config_dir));

    return cat;
}

}  // namespace traccc::opts
