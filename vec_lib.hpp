#ifndef VEC_LIB_HPP_
#define VEC_LIB_HPP_

#include <bit>
#include <cstdint>

#include <immintrin.h>

class Vec8x32f {
  private:
    __m256 vec_;
  public:
    explicit Vec8x32f(const float* f32data) {
        vec_ = _mm256_loadu_ps(f32data);
    }

    explicit Vec8x32f(float val) {
        vec_ = _mm256_set1_ps(val);
    }

    Vec8x32f(const Vec8x32f& other) : vec_{other.vec_} {}
    explicit Vec8x32f(__m256 raw_vec) : vec_{raw_vec} {}
    
    void Load(const float* f32data) {
        vec_ = _mm256_loadu_ps(f32data);
    }

    void Store(float* f32data) {
        _mm256_storeu_ps(f32data, vec_);
    }

    __m256 GetRawVec() {
        return vec_;
    }

    Vec8x32f& operator=(Vec8x32f other) {
        vec_ = other.vec_;
        return *this;
    }

    Vec8x32f operator*(Vec8x32f other) {
        return Vec8x32f{
            _mm256_mul_ps(vec_, other.vec_)
        };
    }

    Vec8x32f operator+(Vec8x32f other) {
        return Vec8x32f{
            _mm256_add_ps(vec_, other.vec_)
        };
    }

    uint32_t operator<=(Vec8x32f other) {
        return std::bit_cast<uint32_t>(
            _mm256_movemask_ps(
                _mm256_cmp_ps(vec_, other.vec_, _CMP_LE_OQ)
            )
        );
    }
};

class Vec4x64u {
  private:
    __m256i vec_;  
  public:
    explicit Vec4x64u(const uint64_t* u64data) {
        vec_ = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u64data));
    }

    explicit Vec4x64u(uint64_t val = 0) {
        vec_ = _mm256_set1_epi64x(std::bit_cast<int64_t>(val));
    }

    Vec4x64u(const Vec4x64u& other) : vec_{other.vec_} {}
    explicit Vec4x64u(__m256i val) : vec_{val} {}

    void Load(const uint64_t* u64data) {
        vec_ = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u64data));
    }

    void Store(uint64_t* u64data) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(u64data), vec_);
    }

    __m256i GetRawVec() {
        return vec_;
    }

    Vec4x64u& operator=(Vec4x64u other) {
        vec_ = other.vec_;
        return *this;
    }

    Vec4x64u operator*(Vec4x64u other) {
        return Vec4x64u{
            _mm256_mul_epu32(vec_, other.vec_)
        };
    }

    Vec4x64u operator+(Vec4x64u other) {
        return Vec4x64u{
            _mm256_add_epi64(vec_, other.vec_)
        };
    }

    Vec4x64u operator>>(uint32_t count) {
        __m128i vec_count = _mm_set1_epi64x(count);
        return Vec4x64u{
            _mm256_srl_epi64(vec_, vec_count)
        };
    }

    Vec4x64u operator&(Vec4x64u other) {
        return Vec4x64u{
            _mm256_and_si256(vec_, other.vec_)
        };
    }
};

class Vec8x32u {
  private:
    __m256i vec_;
  public:
    explicit Vec8x32u(const uint32_t* u32data) {
        vec_ = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u32data));
    }

    explicit Vec8x32u(uint32_t val = 0) {
        vec_ = _mm256_set1_epi32(std::bit_cast<int32_t>(val));
    }

    Vec8x32u(const Vec8x32u& other) : vec_{other.vec_} {}
    explicit Vec8x32u(__m256i val) : vec_{val} {}

    void Load(const uint32_t* u32data) {
        vec_ = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u32data));
    }

    void Store(uint32_t* u32data) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(u32data), vec_);
    }
    
    void Reduce2u64Tou32(Vec4x64u vec_lo, Vec4x64u vec_hi) {
        __m256i perm_mask = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);
        __m128i lo = _mm256_castsi256_si128(
            _mm256_permutevar8x32_epi32(vec_lo.GetRawVec(), perm_mask)
        );
        __m128i hi = _mm256_castsi256_si128(
            _mm256_permutevar8x32_epi32(vec_hi.GetRawVec(), perm_mask)
        );

        vec_ = _mm256_set_m128i(hi, lo);
    }


    __m256i GetRawVec() {
        return vec_;
    }

    Vec8x32u& operator=(Vec8x32u other) {
        vec_ = other.vec_;
        return *this;
    }
};

inline uint32_t CountOnes(uint32_t val) {
    return std::bit_cast<uint32_t>(_mm_popcnt_u32(val));
}

template <typename FromT, typename ToT>
ToT VecBitCast(FromT vec);

template <>
Vec8x32u VecBitCast(Vec8x32f vec) {
    return Vec8x32u{
        _mm256_castps_si256(vec.GetRawVec())
    };
}

template <>
Vec8x32f VecBitCast(Vec8x32u vec) {
    return Vec8x32f{
        _mm256_castsi256_ps(vec.GetRawVec())
    };
}

template <typename FromT, typename ToT>
ToT VecValueCast(FromT vec);

template <>
Vec8x32f VecValueCast(Vec8x32u vec) {
    return Vec8x32f{
        _mm256_cvtepi32_ps(vec.GetRawVec())
    };
}

// FIXME implement casts

#endif // VEC_LIB_HPP_
