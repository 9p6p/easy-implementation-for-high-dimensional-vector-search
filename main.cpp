#include <iostream>
#include <vector>
#include "index.h"

candidate_pool searchBrute(int N, int D, const float* data, const float* query, unsigned topK) {
    candidate_pool candidate_set;
    for (int i = 0; i < N; ++i) {
        float dist = compare(query, data + i * D, D);
        candidate_set.emplace(dist, i);
    }
    while (candidate_set.size() > topK) {
        candidate_set.pop();
    }
    return candidate_set;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    unsigned N, D, k;
    std::cin >> N >> D;
    float* data = new float[N * D];
    float* center = new float[D]();

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            std::cin >> data[i * D + j];
            center[j] += data[i * D + j];
        }
    }
    for (int j = 0; j < D; ++j) {
        center[j] /= N;
    }
    // ... 建图 ...
    Index nsg(D, N, data, center);
    nsg.BuildKNN(N, 10, 10, 10, 8);
    nsg.BuildNSG(10, 10, 20);

    // 输出 "ok"
    if (nsg.has_built)
        std::cout << "ok" << std::endl;

    std::cin >> k;
    float* query = new float[D];
    while (true) {
        for (int j = 0; j < D; ++j) {
            std::cin >> query[j];
            if (std::cin.fail()) {
                return 0;
            }
        }
        candidate_pool tmp;
        // tmp = searchBrute(N, D, data, query, k);
         tmp = nsg.search(query, k, 500);
        // tmp = nsg.searchRandom(query, k, 500);
        int pos = k;
        while (!tmp.empty()) {
            std::cout << tmp.top().second;
            tmp.pop();
            if (pos > 0)
                std::cout << " ";
            pos--;
        }
        std::cout << std::endl;
    }

    delete[] center;
    delete[] query;
    delete[] data;

    return 0;
}
