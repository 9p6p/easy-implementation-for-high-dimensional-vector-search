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
    // 关闭同步
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Timer time;
    time.tick();
    unsigned N, D, k;

    std::cout << "start" << std::endl;
    float* data = read_fbin<float>("/home/yuxiang/src/data-50K", N, D);
    float* center = new float[D];
    for (unsigned j = 0; j < D; j++)
        center[j] = 0;
    for (unsigned i = 0; i < N; i++) {
        for (unsigned j = 0; j < D; j++) {
            center[j] += data[i * D + j];
        }
    }
    for (int i = 0; i < N; i++) {
        normalize(data + i * D, D);
    }
    for (unsigned j = 0; j < D; j++) {
        center[j] /= N;
    }
    // ... 建图 ...
    Index nsg(D, N, data, center);
    nsg.BuildKNN(N, 10, 10, 10, 8);
    time.tuck("build knn done");
    time.tick();
    nsg.BuildNSG(10, 10, 50);

    // 输出 "ok"
    if (nsg.has_built)
        std::cout << "ok" << std::endl;

    time.tuck("build nsg done");

    unsigned nq = 0, gtd = 2;
    k = 10;
    float* query = read_fbin<float>("/home/yuxiang/src/sample-data-800", nq, D);
    for (int i = 0; i < nq; i++) {
        normalize(query + i * D, D);
    }
    unsigned* gt = read_fbin<unsigned>("/home/yuxiang/src/gt-50K-sample-800", nq, gtd);
    time.tick();
    nsg.testSearch(nq, gtd, query, gt, k, 600, true);
    time.tuck("random search done");
    time.tick();
    // nsg.testSearch(nq, gtd, query, gt, k, 500);
    nsg.testSearch(nq, gtd, query, gt, k, 600, false);
    time.tuck("normal search done");

    delete[] center;
    delete[] query;
    delete[] data;

    return 0;
}
