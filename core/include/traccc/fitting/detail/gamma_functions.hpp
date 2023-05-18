/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

namespace traccc::detail {

namespace {
static const double kMACHEP = 1.11022302462515654042363166809e-16;
static const double kMAXLOG = 709.782712893383973096206318587;
static const double kBig = 4.503599627370496e15;
static const double kBiginv = 2.22044604925031308085e-16;

/* Logarithm of gamma function */
/* A[]: Stirling's formula expansion of log gamma
 * B[], C[]: log gamma function between 2 and 3
 */

static double A[] = {8.11614167470508450300E-4, -5.95061904284301438324E-4,
                     7.93650340457716943945E-4, -2.77777777730099687205E-3,
                     8.33333333333331927722E-2};

static double B[] = {-1.37825152569120859100E3, -3.88016315134637840924E4,
                     -3.31612992738871184744E5, -1.16237097492762307383E6,
                     -1.72173700820839662146E6, -8.53555664245765465627E5};

static double C[] = {
    /* 1.00000000000000000000E0, */
    -3.51815701436523470549E2, -1.70642106651881159223E4,
    -2.20528590553854454839E5, -1.13933444367982507207E6,
    -2.53252307177582951285E6, -2.01889141433532773231E6};

}  // namespace

double igamc(double a, double x);

/* left tail of incomplete gamma function:
 *
 *          inf.      k
 *   a  -x   -       x
 *  x  e     >   ----------
 *           -     -
 *          k=0   | (a+k+1)
 *
 */
double igam(double a, double x);

double lgam(double x);

/*
 * calculates a value of a polynomial of the form:
 * a[0]x^N+a[1]x^(N-1) + ... + a[N]
 */
double Polynomialeval(double x, double* a, unsigned int N);

// Funtions to calculate the upper incomplete gamma function from a given chi2
// and ndf
//
// @param x chi square
// @param r ndof
// @return upper incomplete gamma function (pvalue)
template <typename scalar_t>
scalar_t chisquared_cdf_c(scalar_t x, scalar_t r) {
    double retval =
        igamc(0.5 * static_cast<double>(r), 0.5 * static_cast<double>(x));
    return static_cast<scalar_t>(retval);
}

}  // namespace traccc::detail