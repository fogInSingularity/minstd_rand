#ifndef MINSTD_RAND_HPP_
#define MINSTD_RAND_HPP_

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <memory>

#include <immintrin.h>

#include "vec_lib.hpp"

namespace rnd {

class minstd_rand {
  public:
    constexpr static uint64_t default_seed = 1u;
    constexpr static uint64_t multiplier = 48271; // prime
    constexpr static uint64_t modulo = 2147483647; // 2^31 - 1

    constexpr inline static uint32_t min() { return 1; }
    constexpr inline static uint32_t max() { return modulo - 1; }
  private:   
    constexpr static uint64_t a = multiplier;
    constexpr static uint64_t a8 = 854716505; // a^8 mod m
    constexpr static uint64_t mask = 0x7FFFFFFF;
    constexpr static uint64_t shift = 31;

    constexpr static size_t n_ints = 8;
    constexpr static size_t alignment = sizeof(Vec4x64u);

    uint64_t state_;

    inline constexpr uint64_t mod(uint64_t x) {
        uint64_t mul = a * x;
        mul = (mul >> shift) + (mul & mask);
        mul = (mul >> shift) + (mul & mask);
        return mul;
    }

    void gen_to_mem(size_t num, float* gen_val) {
        alignas(alignment) uint64_t lstate[n_ints];
        lstate[0] = state_;
        for (size_t i = 1; i < n_ints; i++) {
            lstate[i] = mod(lstate[i - 1]);
        }

        size_t full_iters = num / n_ints;
        size_t rem_iters = num % n_ints;

        constexpr float norm_mul = 1 / static_cast<float>(max() - min() + 1);

        Vec4x64u state_vec_lo{&lstate[0]};
        Vec4x64u state_vec_hi{&lstate[n_ints / 2]};

        auto mask_vec = Vec4x64u{mask};
        auto a8_vec = Vec4x64u{a8};
        auto norm_vec = Vec8x32f{norm_mul};

        for (size_t j = 0; j < full_iters; j++) {
            // set gen val
            Vec8x32u res_vec{};
            res_vec.Reduce2u64Tou32(state_vec_lo, state_vec_hi);    
            auto fvec = VecValueCast<Vec8x32u, Vec8x32f>(res_vec);
            fvec = fvec * norm_vec;

            fvec.Store(&gen_val[j * n_ints]);

            // update

            state_vec_lo = a8_vec * state_vec_lo;
            state_vec_hi = a8_vec * state_vec_hi;

            state_vec_lo = (state_vec_lo >> shift) + (state_vec_lo & mask_vec);
            state_vec_hi = (state_vec_hi >> shift) + (state_vec_hi & mask_vec);

            state_vec_lo = (state_vec_lo >> shift) + (state_vec_lo & mask_vec);
            state_vec_hi = (state_vec_hi >> shift) + (state_vec_hi & mask_vec);
        }

        Vec8x32u res_vec{};
        res_vec.Reduce2u64Tou32(state_vec_lo, state_vec_hi);

        alignas(alignment) uint32_t rem_vals[n_ints];
        res_vec.Store(rem_vals);
    
        for (size_t i = 0; i < rem_iters; i++) {
            gen_val[full_iters * n_ints + i] = static_cast<float>(rem_vals[i]) * norm_mul;
        }

        state_ = rem_vals[rem_iters];
    }

  public:
    explicit minstd_rand(uint32_t seed = default_seed) : state_{mod(seed)}  {}

    std::unique_ptr<float> operator()(size_t num) {
        float* gen_val = new float[num];

        gen_to_mem(num, gen_val);

        return std::unique_ptr<float>{gen_val};
    }

    void operator()(size_t num, float* gen_val) {
        gen_to_mem(num, gen_val);
    }

    uint32_t operator()() {
        auto state = state_;
        state_ = mod(state_);
        return static_cast<uint32_t>(state);
    }
};

} // namespace rnd

#endif // MINSTD_RAND_HPP_
