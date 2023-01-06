/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <string>

#include "TEfficiency.h"
#include "TFitResult.h"
#include "TFitResultPtr.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TROOT.h"

namespace traccc {

namespace plot_helpers {
/// @brief Nested binning struct for booking plots
struct binning {
    binning();

    binning(std::string b_title, int bins, float b_min, float b_max);

    std::string title;  ///< title to be displayed
    int n_bins;         ///< number of bins
    float min;          ///< minimum value
    float max;          ///< maximum value
};

/// @brief book a 1D histogram
/// @param histName the name of histogram
/// @param histTitle the title of histogram
/// @param varBinning the binning info of variable
/// @return histogram pointer
TH1F* book_histo(const char* hist_name, const char* hist_title,
                 const binning& var_binning);

/// @brief fill a 1D histogram
/// @param hist histogram to fill
/// @param value value to fill
/// @param weight weight to fill
void fill_histo(TH1F* hist, float value, float weight = 1.0);

/// @brief extract details, i.e. mean and width of a 1D histogram and fill
/// them into histograms
/// @param inputHist histogram to investigate
/// @param j  which bin number of meanHist and widthHist to fill
/// @param meanHist histogram to fill the mean value of inputHist
/// @param widthHist  histogram to fill the width value of inputHist
///
/// @todo  write specialized helper class to extract details of hists
void ana_histo(TH1D* input_hist, int j, TH1F* mean_hist, TH1F* width_hist);

/// @brief book a 1D efficiency plot
/// @param effName the name of plot
/// @param effTitle the title of plot
/// @param varBinning the binning info of variable
/// @return TEfficiency pointer
TEfficiency* book_eff(const char* eff_name, const char* eff_title,
                      const binning& var_binning);

/// @brief fill a 1D efficiency plot
/// @param efficiency plot to fill
/// @param value value to fill
/// @param status bool to denote passed or not
void fill_eff(TEfficiency* efficiency, float value, bool status);

TProfile* book_prof(const char* prof_name, const char* prof_title,
                    const binning& var_x_binning, const binning& var_y_binning);

void fill_prof(TProfile* profile, float x_value, float y_value,
               float weight = 1.0);

}  // namespace plot_helpers

}  // namespace traccc
