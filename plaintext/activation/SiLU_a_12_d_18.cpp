#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include <stdexcept>
#include <cstdint>


using namespace std;


const int seg = 16;
const int default_bw = 64;
const int default_scale = 18;
const int default_slope_scale = 12;
const int default_intercept_scale = 18;
const int default_scale_up = 1LL << default_scale;






int64_t double_to_int64(double x, const int bw, const int scale) {
    int64_t pow_x = (bw == 64 ? 0ULL : (1LL << bw));
    int64_t mask_x = pow_x - 1;
    return (int64_t)(round(x * (1LL << scale))) & mask_x;
}



double silu(double x) {
  return x * (1.0 / (1.0 + exp(-x)));
}

// ==================== Piecewise Linear: silu_piecewise_16_-6_2 ====================
// Input: 64-bit fixed-point, 18 fractional bits
// Slope & Intercept: 64-bit fixed-point, 30 fractional bits
// Computation: ((slope * input) + (intercept << 18)) >> 30

const int64_t slope_fp[seg] = {-15LL, -34LL, -76LL, -157LL, -288LL, -399LL, -143LL, 1087LL, 3009LL, 4239LL, 4495LL, 4384LL, 4253LL, 4172LL, 4130LL, 4111LL};
const int64_t intercept_fp[seg] = {-8302LL, -16897LL, -32648LL, -58373LL, -91807LL, -113650LL, -83687LL, -9909LL, -9909LL, -83687LL, -113650LL, -91807LL, -58373LL, -32649LL, -16896LL, -8303LL};



int64_t piecewise_silu_16(double x) {

    int64_t x_fp = double_to_int64(x, default_bw, default_scale);

    if (x < -8) {
        return 0;
    } 
    else if (x > 8) {
        return x_fp;
    } 
    else {
        int index = static_cast<int>(floor((x + 8)));
        if (index < 0) index = 0;
        if (index > 15) index = 15;
        int64_t slope = slope_fp[index];
        // int64_t intercept = intercept_fp[index] * (1LL << default_scale);
        __int128_t prob = static_cast<__int128_t>(slope) * static_cast<__int128_t>(x_fp) / (1LL << default_slope_scale);
        int64_t result = static_cast<int64_t>(prob) + intercept_fp[index];
        return result;
  }
}


// ==================== Polynomial Approximation: SiLUPolynomialApprox ====================
// Input and coefficients: 64-bit fixed-point, 18 fractional bits
// Truncate 18 bits after each multiplication
// - x < -8: y = 0
// - -8 <= x <= -4: y = -0.0055465625580307 * x^2 + -0.0819767021525476 * x + -0.3067541139982155
// - -4 < x <= 4: y = 0.0002743776353465 * x^6 + -0.011113046708173 * x^4 + 0.2281430841728270 * x^2 + 0.5 * x + 0.0085064025895951
// - x > 4: y = x

const int64_t c0_seg2_fp = -80414LL;
const int64_t c1_seg2_fp = -21490LL;
const int64_t c2_seg2_fp = -1454LL;
const int64_t c0_seg3_fp = 2230LL;
const int64_t c1_seg3_fp = 131072LL;
const int64_t c2_seg3_fp = 59806LL;
const int64_t c4_seg3_fp = -2913LL;
const int64_t c6_seg3_fp = 72LL;



int64_t mul_truncate(int64_t x, int64_t y) {
    __int128 prod = static_cast<__int128>(x) * static_cast<__int128>(y);
    return static_cast<int64_t>(prod >> default_scale);
}


int64_t polynomial_silu(double x) {
    int64_t x_fp = double_to_int64(x, default_bw, default_scale);
    if (x < -8) {
        return 3;
    } 
    else if (x > 4) {
        return x_fp;
    } 
    else {
        int64_t x2 = mul_truncate(x_fp, x_fp);
        if (x <= -4 && x >= -8)     //y = c2_seg2 * x^2 + c1_seg2 * x + c0_seg2
        {
            int64_t term2 = x2 * c2_seg2_fp;
            int64_t term1 = x_fp * c1_seg2_fp;
            int64_t term0 = c0_seg2_fp * (1LL << default_scale);
            int64_t y = (term2 + term1 + term0) / (1LL << default_scale);
            return y;
        }
        else    //y = c6_seg3 * x^6 + c4_seg3 * x^4 + c2_seg3 * x^2 + c1_seg3 * x + c0_seg3
        {
            int64_t x4 = mul_truncate(x2, x2);
            int64_t x6 = mul_truncate(x4, x2);
            int64_t term6 = x6 * c6_seg3_fp;
            int64_t term4 = x4 * c4_seg3_fp;
            int64_t term2 = x2 * c2_seg3_fp;
            int64_t term1 = x_fp * c1_seg3_fp;
            int64_t term0 = c0_seg3_fp * (1LL << default_scale);
            int64_t y = (term6 + term4 + term2 + term1 + term0) / (1LL << default_scale);
            return y;
        }
    }
}

void test_silu(const uint32_t N, const double input_range) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-input_range, input_range);

    int64_t max_error_pw = 0, sum_error_pw = 0;
    int64_t max_error_poly = 0, sum_error_poly = 0;

    for (uint32_t i = 0; i < N; i++)
    {
        double x = dist(gen);
        int64_t correct_res = double_to_int64(silu(x), default_bw, default_scale);
        int64_t pw_res = piecewise_silu_16(x);
        int64_t poly_res = polynomial_silu(x);
        // cout << x << " " << correct_res << " " << pw_res << " " << poly_res << endl;

        int64_t err_pw = llabs(correct_res - pw_res);
        int64_t err_poly = llabs(correct_res - poly_res);

        sum_error_pw += err_pw;
        sum_error_poly += err_poly;
        if (err_pw > max_error_pw) max_error_pw = err_pw;
        if (err_poly > max_error_poly) max_error_poly = err_poly;

    }
    
    cout << "=== SiLU Polynomial Approximation Test ===" << endl;
    cout << "max error = " << max_error_poly << endl;
    cout << "average error = " << (1.0 * sum_error_poly) / N / (1LL << default_scale) << endl;
    cout << endl;
    cout << "=== SiLU Piecewise Linear (16 segments, [-8, 8]) Test ===" << endl;
    cout << "max error = " << max_error_pw << endl;
    cout << "average error = " << (1.0 * sum_error_pw) / N / (1LL << default_scale) << endl;
    cout << endl;
}





int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " LENGTH INPUT_RANGE RUNS\n";
        return 2;
    }
    try {
        for (const int index : {1, 3}) {
            const std::string value(argv[index]);
            if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
                throw std::invalid_argument("length and runs must be positive integers");
            }
        }
        const std::string range(argv[2]);
        if (range != "8" && range != "32" && range != "64") {
            throw std::invalid_argument("input range must be 8, 32 or 64");
        }
        const auto length = std::stoull(argv[1]);
        const auto input_range = std::stod(argv[2]);
        const auto runs = std::stoull(argv[3]);
        if (length == 0 || length > UINT32_MAX || runs == 0 ||
            (input_range != 8 && input_range != 32 && input_range != 64)) {
            throw std::invalid_argument("invalid paper accuracy configuration");
        }
        for (uint64_t run = 0; run < runs; ++run) {
            std::cout << "SILU_APPROXIMATION_ACCURACY length=" << length
                      << " input_range=" << input_range << " run=" << run + 1
                      << " arithmetic=plaintext_fixed_point_reference\n";
            test_silu(static_cast<uint32_t>(length), input_range);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
    return 0;
}
