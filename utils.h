#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

typedef std::priority_queue<std::pair<float, unsigned>> candidate_pool;

void GenRandom(std::mt19937& rng, unsigned* addr, size_t size, size_t N) {
    for (size_t i = 0; i < size; ++i) {
        addr[i] = rng() % (N - size);
    }
    std::sort(addr, addr + size);
    for (size_t i = 1; i < size; ++i) {
        if (addr[i] <= addr[i - 1]) {
            addr[i] = addr[i - 1] + 1;
        }
    }
    size_t off = rng() % N;
    for (size_t i = 0; i < size; ++i) {
        addr[i] = (addr[i] + off) % N;
    }
}

inline float compare(const float* a, const float* b, unsigned dim) {
    float dist = 0;
    for (unsigned i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

struct Timer {
    std::chrono::_V2::system_clock::time_point s;
    std::chrono::_V2::system_clock::time_point e;
    std::chrono::duration<double> diff;

    void tick() {
        s = std::chrono::high_resolution_clock::now();
    }

    void tuck(std::string message) {
        e = std::chrono::high_resolution_clock::now();
        diff = e - s;
        std::cout << "[" << diff.count() << " s] " << message << std::endl;
    }
};

void normalize(float* arr, unsigned dim) {
    float sum = 0.0f;
    for (unsigned i = 0; i < dim; i++) {
        sum += arr[i] * arr[i];
    }
    sum = sqrt(sum);
    for (unsigned i = 0; i < dim; i++) {
        arr[i] = arr[i] / sum;
    }
}

template <class T>
T* read_fbin(const char* filename, unsigned& n, unsigned& d) {
    std::ifstream in(filename, std::ios::binary);
    in.read((char*)&n, 4);
    in.read((char*)&d, 4);
    T* data = new T[n * d];
    // std::cout << n << " " << d << std::endl;
    in.read((char*)data, (size_t)n * (size_t)d * 4);
    in.close();
    return data;
}

#endif