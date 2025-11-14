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
    m_desc.add_options()("useGBTS",
                         boost::program_options::bool_switch(&useGBTS),
                         "use gbts algorithm");

    m_desc.add_options()(
        "gbts_config_dir",
        boost::program_options::value(&config_dir)->default_value(config_dir),
        "directory for gbts config files");
}

void track_gbts_seeding::read(const boost::program_options::variables_map &) {
    // fill config
    if (!useGBTS) {
        return;
    }
    std::ifstream barcodeBinningFile(
        std::filesystem::path(config_dir + "/barcodeBinning.txt"));

    int nBarcodes = 0;
    barcodeBinningFile >> nBarcodes;
    barcodeBinning.reserve(nBarcodes);

    std::pair<uint64_t, short> barcodeLayerPair;
    for (; nBarcodes > 0; --nBarcodes) {
        barcodeBinningFile >> barcodeLayerPair.first;
        barcodeBinningFile >> barcodeLayerPair.second;

        barcodeBinning.push_back(barcodeLayerPair);
    }

    std::ifstream binTablesFile(
        std::filesystem::path(config_dir + "/binTables.txt"));

    int nBinPairs = 0;
    binTablesFile >> nBinPairs;
    binTables.reserve(nBinPairs);
    int bin1 = 0;
    std::vector<int> bin2 = {0};
    for (; nBinPairs > 0; --nBinPairs) {
        binTablesFile >> bin1;
        binTablesFile >> bin2[0];
        binTables.push_back(std::make_pair(bin1, bin2));
    }

    std::ifstream layerInfoFile(
        std::filesystem::path(config_dir + "/layerInfo.txt"));

    int nLayers = 0;
    layerInfoFile >> nLayers;
    layerInfo.reserve(nLayers);
    char type = 0;
    int info[2] = {0, 0};
    float geo[2] = {0, 0};
    for (; nLayers > 0; --nLayers) {
        layerInfoFile >> type;
        layerInfoFile >> info[0] >> info[1];
        layerInfoFile >> geo[0] >> geo[1];
        layerInfo.addLayer(type, info[0], info[1], geo[0], geo[1]);
    }
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
