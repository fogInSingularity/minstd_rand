# minstd_rand
optimized std::minstd_rand algorithm

## rnd::minstd_rand API

```cpp
// UniformRandomBitGenerator compient interface:
rnd::minstd_rand::minstd_rand(seed = default_seed);
uint32_t rnd::minstd_rand::max();
uint32_t rnd::minstd_rand::min();
uint32_t rnd::minstd_rand::operator();
// extended interface for generating chuncks of random uniformed distributed float values:
std::unique_ptr<float> rnd::minstd_rand::operator()(size_t num);
void rnd::minstd_rand::operator()(size_t num, float* gen_val);
```

## Monte Carlo Pi benchmarking

### Benchmark implementation:

`std::minstd_rand` version:

```cpp
size_t points_inside_cirl = 0;

std::minstd_rand rgen{rseed};
std::uniform_real_distribution<float> udistr{0.0, 1.0};
for (size_t i = 0; i < kIters; i++) {
    float x = udistr(rgen);
    float y = udistr(rgen);
    if (x * x + y * y <= 1) {
        points_inside_cirl++;
    }
}

float pi = points_inside_cirl / total_num_points;
```

`rnd::minstd_rand` version

```cpp
size_t points_inside_cirl = 0;

rnd::minstd_rand rgen{rseed};

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

float pi = points_inside_cirl / total_num_points;
```

### Benchmark parameters

| | |
|-|-|
| cpu architecture | x86-64 |
| cpu | AMD Ryzen 5 3500u (Zen+) |
| os | Linux fedora-41 kernel 6.14.3 |
| compiler | gcc 14.2.1 | 
| compiler options | -O2 -march=native |
| `kIters` | 10^9 | 

### Benchmarking results

for benchmarking i used hyperfine:

```bash
$ hyperfine -w 5 -r 20 "./build/minstd_rand >> log"
Benchmark 1: ./build/minstd_rand >> log
  Time (mean ± σ):      3.124 s ±  0.064 s    [User: 23.151 s, System: 0.019 s]
  Range (min … max):    3.006 s …  3.238 s    20 runs
```

```bash
$ hyperfine -w 5 -r 20 "./build/minstd_rand >> log"
Benchmark 1: ./build/minstd_rand >> log
  Time (mean ± σ):     11.921 s ±  0.283 s    [User: 88.794 s, System: 0.066 s]
  Range (min … max):   11.640 s … 12.927 s    20 runs
```

`rnd::minstd_rand` faster than `std::minstd_rand` faster by 3.8 times on this benchmark
