#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <cstddef>
#include <ostream>
#include <popcntintrin.h>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <immintrin.h>

#include "minstd_rand.hpp"
#include "vec_lib.hpp"

void CountOnThread(bool is_naive, unsigned rseed, size_t* points_in_cirl);

// result: 3.14161 (should be 3.14159)
constexpr size_t kIters = 100'000'000;

int main(const int argc, const char** argv) {
    if (argc < 3) {
        std::cerr << "not enough args" << std::endl;
        std::cerr << argv[0] << " <number_of_threads> [naive|vec]" << std::endl;
        
        return EXIT_FAILURE;
    }

    size_t n_threads = 0;
    try {
        n_threads = static_cast<size_t>(std::stoll(argv[1]));
    } catch (const std::exception& e) {
        std::cout << "Unable to get number of theads: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    bool is_naive = false;
    if (argv[2] == std::string{"naive"}) {
        is_naive = true;
    } else if (argv[2] == std::string{"vec"}) {
        is_naive = false;
    } else {
        std::cerr << argv[0] << " <number_of_threads> [naive|vec]" << std::endl;
        
        return EXIT_FAILURE;
    }

    std::vector<unsigned> seed_array{};
    seed_array.resize(n_threads);

    std::vector<size_t> points_array{};
    points_array.resize(n_threads);

    std::random_device rdevice{};
    for (auto&& e: seed_array) {
        e = rdevice();
    }

    std::vector<std::thread> thread_array{};
    thread_array.reserve(n_threads);
    for (size_t i = 0; i < n_threads; i++) {
        thread_array.emplace_back(CountOnThread, is_naive, seed_array[i], &points_array[i]);
    }

    for (size_t i = 0; i < n_threads; i++) {
        thread_array[i].join();
    }

    size_t points_inside_cirl = 0;
    for (auto&& e: points_array) {
        points_inside_cirl += e;
    }

    size_t number_of_points = kIters * n_threads;
    double pi = 4 * static_cast<double>(points_inside_cirl) / static_cast<double>(number_of_points);

    std::cout << "pi: " << pi << std::endl;
}

void CountOnThread(bool is_naive, unsigned rseed, size_t* points_in_cirl) {
    size_t points_inside_cirl = 0;

    constexpr size_t vec_size = 8;
    constexpr size_t chuck = 10'000;
    constexpr size_t sub_iters = kIters / chuck;

    float* val_ptr = new float[2 * chuck];

    if (is_naive) {
        thread_local std::minstd_rand rgen{rseed};
        thread_local std::uniform_real_distribution<float> udistr{0.0, 1.0};

        for (size_t outer_ind = 0; outer_ind < sub_iters; outer_ind++) {
            for (size_t i = 0; i < 2 * chuck; i++) {
                val_ptr[i] = udistr(rgen);
            }

            for (size_t i = 0; i < 2 * chuck; i += 2 * vec_size) {
                Vec8x32f vec_x{&val_ptr[i]};
                Vec8x32f vec_y{&val_ptr[i + vec_size]};

                Vec8x32f sqr = vec_x * vec_x + vec_y * vec_y;
                Vec8x32f one_vec = Vec8x32f{1.0f};

                uint32_t cmp_mask = sqr <= one_vec;
                points_inside_cirl += CountOnes(cmp_mask);
            }
        }
    } else {
        thread_local rnd::minstd_rand rgen{rseed};

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
    }

    delete[] val_ptr;

    *points_in_cirl = points_inside_cirl;
}

