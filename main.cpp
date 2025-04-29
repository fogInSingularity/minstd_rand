#include <array>
#include <cstdint>
#include <iostream>
#include <cstddef>
#include <popcntintrin.h>
#include <random>
#include <thread>

#include <immintrin.h>

#include "minstd_rand.hpp"
#include "vec_lib.hpp"

void CountOnThread(unsigned rseed, size_t* points_in_cirl);

// result: 3.14161 (should be 3.14159)
constexpr size_t kIters = 1'000'000'000;
constexpr size_t kNThreads = 8;

int main() {
    std::random_device rdevice{};
    std::array<unsigned, kNThreads> seed_array{};
    std::array<size_t, kNThreads> points_array{};

    for (auto&& e: seed_array) {
        e = rdevice();
    }

    std::array<std::thread, kNThreads> thread_array;
    for (size_t i = 0; i < kNThreads; i++) {
        thread_array[i] = std::thread(CountOnThread, seed_array[i], &points_array[i]);
    }

    for (size_t i = 0; i < kNThreads; i++) {
        thread_array[i].join();
    }

    size_t points_inside_cirl = 0;
    for (auto&& e: points_array) {
        points_inside_cirl += e;
    }

    size_t number_of_points = kIters * kNThreads;
    double pi = 4 * static_cast<double>(points_inside_cirl) / static_cast<double>(number_of_points);

    std::cout << "pi: " << pi << std::endl;
}

void CountOnThread(unsigned rseed, size_t* points_in_cirl) {
    size_t points_inside_cirl = 0;

#if 0
    thread_local std::minstd_rand rgen{rseed};
    thread_local std::uniform_real_distribution<float> udistr{0.0, 1.0};
    for (size_t i = 0; i < kIters; i++) {
        float x = udistr(rgen);
        float y = udistr(rgen);
        if (x * x + y * y <= 1) {
            points_inside_cirl++;
        }
    }   
#else 
    thread_local rnd::minstd_rand rgen{rseed};

    constexpr size_t vec_size = 8;
    constexpr size_t chuck = 10'000;
    constexpr size_t sub_iters = kIters / chuck;

    float* val_ptr = new float[2 * chuck];

    for (size_t outer_ind = 0; outer_ind < sub_iters; outer_ind++) {
        rgen(2 * chuck, val_ptr);

        for (size_t i = 0; i < 2 * chuck; i += 2 * vec_size) {
            Vec8x32f vec_x{&val_ptr[i]};
            Vec8x32f vec_y{&val_ptr[i + vec_size]};

            Vec8x32f sqr = vec_x * vec_x + vec_y * vec_y;
            Vec8x32f one_vec = Vec8x32f{1.0f};

            uint32_t cmp_mask = sqr <= one_vec;
            points_inside_cirl += CountOnes(cmp_mask);
        }
    }

    delete[] val_ptr;
#endif // 0

    *points_in_cirl = points_inside_cirl;
}

