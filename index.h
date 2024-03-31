#ifndef INDEX_H
#define INDEX_H

#include <bitset>
#include <cstring>
#include <stack>
#include "utils.h"

const int MAX_ELEMENTS = 50000;

struct Neighbor {
    unsigned id;
    float distance;
    bool flag;

    Neighbor() = default;

    explicit Neighbor(unsigned id, float distance, bool flag)
        : id{id}, distance{distance}, flag(flag) {}

    inline bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

struct nnNeighbor {
    unsigned M;
    std::vector<Neighbor> pool;
    std::vector<unsigned> nn_old;
    std::vector<unsigned> nn_new;
    std::vector<unsigned> rnn_old;
    std::vector<unsigned> rnn_new;

    explicit nnNeighbor(unsigned size, unsigned L, std::mt19937& rng, size_t N) {
        M = size;
        pool.reserve(L);
        nn_new.resize(size * 2);
        GenRandom(rng, nn_new.data(), size * 2, N);
    }

    void insert(unsigned id, float dist) {
        if (dist > pool.front().distance) {
            return;
        }
        for (unsigned i = 0; i < pool.size(); i++) {
            if (id == pool[i].id) {
                return;
            }
        }
        if (pool.size() < pool.capacity()) {
            pool.push_back(Neighbor(id, dist, true));
            std::push_heap(pool.begin(), pool.end());
        } else {
            std::pop_heap(pool.begin(), pool.end());
            pool[pool.size() - 1] = Neighbor(id, dist, true);
            std::push_heap(pool.begin(), pool.end());
        }
    }

    template <typename C>
    void join(C callback) const {
        for (unsigned const i : nn_new) {
            for (unsigned const j : nn_new) {
                if (i < j) {
                    callback(i, j);
                }
            }
            for (unsigned j : nn_old) {
                callback(i, j);
            }
        }
    }
};

struct SimpleNeighbor {
    unsigned id;
    float distance;

    SimpleNeighbor() = default;

    explicit SimpleNeighbor(unsigned id, float distance)
        : id{id}, distance{distance} {}

    inline bool operator<(const SimpleNeighbor& other) const {
        return distance < other.distance;
    }
};

struct SimpleNeighbors {
    std::vector<SimpleNeighbor> pool;
};

static inline int InsertIntoPool(Neighbor* addr, unsigned K, Neighbor nn) {
    int left = 0, right = K - 1;
    if (addr[left].distance > nn.distance) {
        memmove((char*)&addr[left + 1], &addr[left], K * sizeof(Neighbor));
        addr[left] = nn;
        return left;
    }
    if (addr[right].distance < nn.distance) {
        addr[K] = nn;
        return K;
    }
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (addr[mid].distance > nn.distance)
            right = mid;
        else
            left = mid;
    }

    while (left > 0) {
        if (addr[left].distance < nn.distance)
            break;
        if (addr[left].id == nn.id)
            return K + 1;
        left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id)
        return K + 1;
    memmove((char*)&addr[right + 1], &addr[right],
            (K - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
}

class Index {
   public:
    explicit Index(unsigned dim, const size_t nd, const float* data, const float* center)
        : dim_(dim), nd_(nd), data_(data), center_(center), has_built(false) {}

    inline bool HasBuilt() const { return has_built; }

    unsigned dim_;
    size_t nd_;
    size_t ep_ = 0;
    bool has_built;

    const float* data_;
    const float* center_;

    std::vector<nnNeighbor> knn_graph_;
    std::vector<std::vector<unsigned>> final_graph_;

    unsigned _CONTROL_NUM = 10;
    void generate_control_set(std::vector<unsigned>& c, std::vector<std::vector<unsigned>>& v, unsigned N) {
        for (unsigned i = 0; i < c.size(); i++) {
            std::vector<Neighbor> tmp;
            for (unsigned j = 0; j < N; j++) {
                float dist = compare(data_ + c[i] * dim_, data_ + j * dim_, dim_);
                tmp.push_back(Neighbor(j, dist, true));
            }
            std::partial_sort(tmp.begin(), tmp.begin() + _CONTROL_NUM, tmp.end());
            for (unsigned j = 0; j < _CONTROL_NUM; j++) {
                v[i].push_back(tmp[j].id);
            }
        }
    }
    void eval_recall(std::vector<unsigned>& ctrl_points, std::vector<std::vector<unsigned>>& acc_eval_set) {
        float mean_acc = 0;
        for (unsigned i = 0; i < ctrl_points.size(); i++) {
            float acc = 0;
            auto& g = knn_graph_[ctrl_points[i]].pool;
            auto& v = acc_eval_set[i];
            for (unsigned j = 0; j < g.size(); j++) {
                for (unsigned k = 0; k < v.size(); k++) {
                    if (g[j].id == v[k]) {
                        acc++;
                        break;
                    }
                }
            }
            mean_acc += acc / v.size();
        }
        std::cout << "recall : " << mean_acc / ctrl_points.size() << std::endl;
    }

    void join() {
        for (unsigned n = 0; n < nd_; ++n) {
            knn_graph_[n].join([&](unsigned i, unsigned j) {
                if (i != j) {
                    float dist = compare(data_ + i * dim_, data_ + j * dim_, dim_);
                    knn_graph_[i].insert(j, dist);
                    knn_graph_[j].insert(i, dist);
                }
            });
        }
    }

    void update(unsigned size, unsigned L) {
        for (unsigned i = 0; i < nd_; ++i) {
            std::vector<unsigned>().swap(knn_graph_[i].nn_new);
            std::vector<unsigned>().swap(knn_graph_[i].nn_old);
        }
        for (unsigned n = 0; n < nd_; ++n) {
            auto& nn = knn_graph_[n];
            std::sort(nn.pool.begin(), nn.pool.end());
            if (nn.pool.size() > L)
                nn.pool.resize(L);
            unsigned maxl = std::min(nn.M + size, (unsigned)nn.pool.size());
            unsigned c = 0;
            unsigned l = 0;
            while ((l < maxl) && (c < size)) {
                if (nn.pool[l].flag)
                    ++c;
                ++l;
            }
            nn.M = l;
        }
        for (unsigned n = 0; n < nd_; ++n) {
            auto& nnhd = knn_graph_[n];
            auto& nn_new = nnhd.nn_new;
            auto& nn_old = nnhd.nn_old;
            for (unsigned l = 0; l < nnhd.M; ++l) {
                auto& nn = nnhd.pool[l];
                auto& nhood_o = knn_graph_[nn.id];  // nn on the other side of the edge
                if (nn.flag) {
                    nn_new.push_back(nn.id);
                    if (nn.distance > nhood_o.pool.back().distance) {
                        if (nhood_o.rnn_new.size() < size)
                            nhood_o.rnn_new.push_back(n);
                        else {
                            unsigned int pos = rand() % size;
                            nhood_o.rnn_new[pos] = n;
                        }
                    }
                    nn.flag = false;
                } else {
                    nn_old.push_back(nn.id);
                    if (nn.distance > nhood_o.pool.back().distance) {
                        if (nhood_o.rnn_old.size() < size)
                            nhood_o.rnn_old.push_back(n);
                        else {
                            unsigned int pos = rand() % size;
                            nhood_o.rnn_old[pos] = n;
                        }
                    }
                }
            }
            std::make_heap(nnhd.pool.begin(), nnhd.pool.end());
        }
        for (unsigned i = 0; i < nd_; ++i) {
            auto& nn_new = knn_graph_[i].nn_new;
            auto& nn_old = knn_graph_[i].nn_old;
            auto& rnn_new = knn_graph_[i].rnn_new;
            auto& rnn_old = knn_graph_[i].rnn_old;
            if (size && rnn_new.size() > size) {
                std::random_shuffle(rnn_new.begin(), rnn_new.end());
                rnn_new.resize(size);
            }
            nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
            if (size && rnn_old.size() > size) {
                std::random_shuffle(rnn_old.begin(), rnn_old.end());
                rnn_old.resize(size);
            }
            nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
            if (nn_old.size() > size * 2) {
                nn_old.resize(size * 2);
            }
            std::vector<unsigned>().swap(knn_graph_[i].rnn_new);
            std::vector<unsigned>().swap(knn_graph_[i].rnn_old);
        }
    }

    void BuildKNN(size_t n, unsigned k, unsigned size, unsigned L, unsigned iter) {
        // std::cout << "k: " << k << " size: " << size << " L: " << L << " iter: " << iter << std::endl;
        knn_graph_.reserve(nd_);
        std::mt19937 rng(rand());
        for (unsigned i = 0; i < nd_; ++i) {
            knn_graph_.push_back(nnNeighbor(size, L, rng, nd_));
            std::vector<unsigned> tmp(size);
            GenRandom(rng, tmp.data(), size, nd_);
            for (unsigned j = 0; j < size; j++) {
                unsigned id = tmp[j];
                if (id == i)
                    continue;
                float dist = compare(data_ + i * dim_, data_ + id * dim_, dim_);
                knn_graph_[i].pool.push_back(Neighbor(id, dist, true));
            }
            std::make_heap(knn_graph_[i].pool.begin(), knn_graph_[i].pool.end());
        }
        // std::vector<unsigned> control_points(_CONTROL_NUM);
        // std::vector<std::vector<unsigned>> acc_eval_set(_CONTROL_NUM);
        // GenRandom(rng, &control_points[0], control_points.size(), nd_);
        // generate_control_set(control_points, acc_eval_set, nd_);
        for (unsigned it = 0; it < iter; it++) {
            join();
            update(size, L);
            // eval_recall(control_points, acc_eval_set);
            // std::cout << "iter: " << it << std::endl;
        }
        final_graph_.reserve(nd_);
        for (unsigned i = 0; i < nd_; i++) {
            std::vector<unsigned> tmp(k);
            std::sort(knn_graph_[i].pool.begin(), knn_graph_[i].pool.end());
            for (unsigned j = 0; j < k; j++) {
                if (knn_graph_[i].pool[j].id < nd_)
                    tmp.push_back(knn_graph_[i].pool[j].id);
            }
            final_graph_.push_back(tmp);
            std::vector<Neighbor>().swap(knn_graph_[i].pool);
            std::vector<unsigned>().swap(knn_graph_[i].nn_new);
            std::vector<unsigned>().swap(knn_graph_[i].nn_old);
            std::vector<unsigned>().swap(knn_graph_[i].rnn_new);
            std::vector<unsigned>().swap(knn_graph_[i].rnn_new);
        }
        std::vector<nnNeighbor>().swap(knn_graph_);
        candidate_pool res = search(center_, 1, 30);
        ep_ = !res.empty() ? res.top().second : rng() % n;
        // std::cout << "ep: " << ep_ << std::endl;
        has_built = true;
    }

    void prune(unsigned R, unsigned C, unsigned q, std::vector<Neighbor>& pool, std::bitset<MAX_ELEMENTS>& flags, SimpleNeighbor* cut_graph_) {
        unsigned start = 0;

        for (unsigned nn = 0; nn < final_graph_[q].size(); nn++) {
            unsigned id = final_graph_[q][nn];
            if (flags[id])
                continue;
            float dist = compare(data_ + dim_ * (size_t)q, data_ + dim_ * (size_t)id, dim_);
            pool.push_back(Neighbor(id, dist, true));
        }

        std::sort(pool.begin(), pool.end());
        std::vector<Neighbor> result;
        if (pool[start].id == q)
            start++;
        result.push_back(pool[start]);

        while (result.size() < R && (++start) < pool.size() && start < C) {
            auto& p = pool[start];
            bool occlude = false;
            for (unsigned t = 0; t < result.size(); t++) {
                if (p.id == result[t].id) {
                    occlude = true;
                    break;
                }
                float djk = compare(data_ + dim_ * (size_t)result[t].id, data_ + dim_ * (size_t)p.id, dim_);
                if (djk < p.distance /* dik */) {
                    occlude = true;
                    break;
                }
            }
            if (!occlude)
                result.push_back(p);
        }

        SimpleNeighbor* des_pool = cut_graph_ + (size_t)q * (size_t)R;
        for (size_t t = 0; t < result.size(); t++) {
            des_pool[t].id = result[t].id;
            des_pool[t].distance = result[t].distance;
        }
        if (result.size() < R) {
            des_pool[result.size()].distance = -1;
        }
    }

    void InterInsert(unsigned n, unsigned range, SimpleNeighbor* cut_graph_) {
        SimpleNeighbor* src_pool = cut_graph_ + (size_t)n * (size_t)range;
        for (size_t i = 0; i < range; i++) {
            if (src_pool[i].distance == -1)
                break;
            SimpleNeighbor sn(n, src_pool[i].distance);
            size_t des = src_pool[i].id;
            SimpleNeighbor* des_pool = cut_graph_ + des * (size_t)range;
            std::vector<SimpleNeighbor> temp_pool;
            int dup = 0;
            for (size_t j = 0; j < range; j++) {
                if (des_pool[j].distance == -1)
                    break;
                if (n == des_pool[j].id) {
                    dup = 1;
                    break;
                }
                temp_pool.push_back(des_pool[j]);
            }
            if (dup)
                continue;
            temp_pool.push_back(sn);
            if (temp_pool.size() > range) {
                std::vector<SimpleNeighbor> result;
                unsigned start = 0;
                std::sort(temp_pool.begin(), temp_pool.end());
                result.push_back(temp_pool[start]);
                while (result.size() < range && (++start) < temp_pool.size()) {
                    auto& p = temp_pool[start];
                    bool occlude = false;
                    for (unsigned t = 0; t < result.size(); t++) {
                        if (p.id == result[t].id) {
                            occlude = true;
                            break;
                        }
                        float djk = compare(data_ + dim_ * (size_t)result[t].id,
                                            data_ + dim_ * (size_t)p.id,
                                            (unsigned)dim_);
                        if (djk < p.distance /* dik */) {
                            occlude = true;
                            break;
                        }
                    }
                    if (!occlude)
                        result.push_back(p);
                }
                {
                    for (unsigned t = 0; t < result.size(); t++) {
                        des_pool[t] = result[t];
                    }
                }
            } else {
                for (unsigned t = 0; t < range; t++) {
                    if (des_pool[t].distance == -1) {
                        des_pool[t] = sn;
                        if (t + 1 < range)
                            des_pool[t + 1].distance = -1;
                        break;
                    }
                }
            }
        }
    }

    void DFS(std::bitset<MAX_ELEMENTS>& flag, unsigned root, unsigned& cnt) {
        unsigned tmp = root;
        std::stack<unsigned> s;
        s.push(root);
        if (!flag[root])
            cnt++;
        flag[root] = true;
        while (!s.empty()) {
            unsigned next = nd_ + 1;
            for (unsigned i = 0; i < final_graph_[tmp].size(); i++) {
                if (flag[final_graph_[tmp][i]] == false) {
                    next = final_graph_[tmp][i];
                    break;
                }
            }
            if (next == (nd_ + 1)) {
                s.pop();
                if (s.empty())
                    break;
                tmp = s.top();
                continue;
            }
            tmp = next;
            flag[tmp] = true;
            s.push(tmp);
            cnt++;
        }
    }

    void findroot(std::bitset<MAX_ELEMENTS>& flag, unsigned& root, unsigned L) {
        unsigned id = nd_;
        for (unsigned i = 0; i < nd_; i++) {
            if (flag[i] == false) {
                id = i;
                break;
            }
        }
        if (id == nd_)
            return;
        std::vector<unsigned> pool;
        candidate_pool tmp = search(data_ + dim_ * id, 10, 40);
        pool.reserve(tmp.size());
        while (!tmp.empty()) {
            pool.emplace_back(tmp.top().second);
            tmp.pop();
        }
        unsigned found = 0;
        for (unsigned i = 0; i < pool.size(); i++) {
            if (flag[pool[i]]) {
                root = pool[i];
                found = 1;
                break;
            }
        }
        if (found == 0) {
            while (true) {
                unsigned rid = rand() % nd_;
                if (flag[rid]) {
                    root = rid;
                    break;
                }
            }
        }
        final_graph_[root].push_back(id);
    }

    void tree_grow(unsigned L) {
        unsigned root = ep_;
        std::bitset<MAX_ELEMENTS> flags;
        unsigned unlinked_cnt = 0;
        while (unlinked_cnt < nd_) {
            DFS(flags, root, unlinked_cnt);
            if (unlinked_cnt >= nd_)
                break;
            findroot(flags, root, L);
        }
    }

    void get_neighbors(const float* query, std::bitset<MAX_ELEMENTS>& flags, std::vector<Neighbor>& retset, std::vector<Neighbor>& fullset, unsigned L) {
        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);

        L = 0;
        for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
            init_ids[i] = final_graph_[ep_][i];
            flags[init_ids[i]] = true;
            L++;
        }
        while (L < init_ids.size()) {
            unsigned id = rand() % nd_;
            if (flags[id])
                continue;
            init_ids[L] = id;
            L++;
            flags[id] = true;
        }

        L = 0;
        for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            if (id >= nd_)
                continue;
            float dist = compare(data_ + dim_ * id, query, dim_);
            retset[i] = Neighbor(id, dist, true);
            fullset.push_back(retset[i]);
            L++;
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int)L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
                    unsigned id = final_graph_[n][m];
                    if (flags[id])
                        continue;
                    flags[id] = true;

                    float dist = compare(query, data_ + dim_ * id, dim_);
                    Neighbor nn(id, dist, true);
                    fullset.push_back(nn);
                    if (dist >= retset[L - 1].distance)
                        continue;
                    int r = InsertIntoPool(retset.data(), L, nn);

                    if (L + 1 < retset.size())
                        ++L;
                    if (r < nk)
                        nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
    }

    void BuildNSG(unsigned L, unsigned R, unsigned C) {
        // std::cout << "L: " << L << " R: " << R << " C: " << C << std::endl;
        has_built = false;
        SimpleNeighbor* cut_graph_ = new SimpleNeighbor[nd_ * (size_t)R];
        std::vector<Neighbor> pool, tmp;
        std::bitset<MAX_ELEMENTS> flags;
        for (unsigned n = 0; n < nd_; ++n) {
            pool.clear();
            tmp.clear();
            flags.reset();
            get_neighbors(data_ + dim_ * n, flags, tmp, pool, L);
            prune(R, C, n, pool, flags, cut_graph_);
        }
        for (unsigned n = 0; n < nd_; ++n) {
            InterInsert(n, R, cut_graph_);
        }
        final_graph_.resize(nd_);
        for (size_t i = 0; i < nd_; i++) {
            SimpleNeighbor* pool = cut_graph_ + i * R;
            unsigned pool_size = 0;
            for (unsigned j = 0; j < R; j++) {
                if (pool[j].distance == -1)
                    break;
                pool_size = j;
            }
            pool_size++;
            final_graph_[i].resize(pool_size);
            for (unsigned j = 0; j < pool_size; j++) {
                final_graph_[i][j] = pool[j].id;
            }
        }
        tree_grow(L);

        /*unsigned max = 0, min = 1e6, avg = 0;
        for (size_t i = 0; i < nd_; i++) {
            auto size = final_graph_[i].size();
            max = max < size ? size : max;
            min = min > size ? size : min;
            avg += size;
        }
        avg /= 1.0 * nd_;
        std::cout << "Degree Statistics: Max = " << max << "  Min = " << min << "  Avg = " << avg << std::endl;*/

        has_built = true;
        delete[] cut_graph_;
    }

    candidate_pool search(const float* query, unsigned k, unsigned L) {
        // std::cout << "start search." << std::endl;
        std::bitset<MAX_ELEMENTS> visited;
        candidate_pool top_candidates;
        candidate_pool candidate_set;

        float dist = compare(query, data_ + ep_ * dim_, dim_);
        top_candidates.emplace(dist, ep_);
        candidate_set.emplace(-dist, ep_);
        visited[ep_] = true;

        while (!candidate_set.empty()) {
            auto current_node_pair = candidate_set.top();
            candidate_set.pop();
            if ((-current_node_pair.first) > top_candidates.top().first && top_candidates.size() == L)
                break;
            for (unsigned m = 0; m < final_graph_[current_node_pair.second].size(); m++) {
                unsigned candidate_id = final_graph_[current_node_pair.second][m];
                if (visited[candidate_id])
                    continue;
                visited[candidate_id] = true;
                float dist = compare(query, data_ + candidate_id * dim_, dim_);
                if (top_candidates.size() < L || dist < top_candidates.top().first) {
                    candidate_set.emplace(-dist, candidate_id);
                    top_candidates.emplace(dist, candidate_id);
                    if (top_candidates.size() > L)
                        top_candidates.pop();
                }
            }
        }
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        return top_candidates;
    }

    candidate_pool searchRandom(const float* query, unsigned k, unsigned L) {
        // std::cout << "start search." << std::endl;
        std::bitset<MAX_ELEMENTS> visited;
        candidate_pool top_candidates;
        candidate_pool candidate_set;

        int num = L;
        float dist = compare(query, data_ + ep_ * dim_, dim_);
        top_candidates.emplace(dist, ep_);
        candidate_set.emplace(-dist, ep_);
        visited[ep_] = true;
        num--;
        for (unsigned i = 0; i < num && i < final_graph_[ep_].size(); i++) {
            unsigned id = final_graph_[ep_][i];
            float dist = compare(query, data_ + id * dim_, dim_);
            visited[id] = true;
            if (top_candidates.size() < L || dist < top_candidates.top().first) {
                candidate_set.emplace(-dist, id);
                top_candidates.emplace(dist, id);
                if (top_candidates.size() > L)
                    top_candidates.pop();
            }
            num--;
        }
        while (num > 0) {
            unsigned id = rand() % nd_;
            if (visited[id])
                continue;
            float dist = compare(query, data_ + id * dim_, dim_);
            visited[id] = true;
            if (top_candidates.size() < L || dist < top_candidates.top().first) {
                candidate_set.emplace(-dist, id);
                top_candidates.emplace(dist, id);
                if (top_candidates.size() > L)
                    top_candidates.pop();
                num--;
            }
        }
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (!candidate_set.empty()) {
            auto current_node_pair = candidate_set.top();
            candidate_set.pop();
            if ((-current_node_pair.first) > top_candidates.top().first && top_candidates.size() == L)
                break;
            for (unsigned m = 0; m < final_graph_[current_node_pair.second].size(); m++) {
                unsigned candidate_id = final_graph_[current_node_pair.second][m];
                if (visited[candidate_id])
                    continue;
                visited[candidate_id] = true;
                float dist = compare(query, data_ + candidate_id * dim_, dim_);
                if (top_candidates.size() < L || dist < top_candidates.top().first) {
                    candidate_set.emplace(-dist, candidate_id);
                    top_candidates.emplace(dist, candidate_id);
                    if (top_candidates.size() > L)
                        top_candidates.pop();
                }
            }
        }
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        return top_candidates;
    }

    void testSearch(unsigned nq, unsigned k, const float* query, unsigned* gt, unsigned topK, unsigned L, bool random) {
        std::vector<candidate_pool> knn(nq);
        if (random)
            for (size_t i = 0; i < nq; i++) {
                knn[i] = searchRandom(query + i * dim_, topK, L);
            }
        else
            for (size_t i = 0; i < nq; i++) {
                knn[i] = search(query + i * dim_, topK, L);
            }
        float recall = 0;
        std::vector<int> res(11);
        for (size_t i = 0; i < nq; i++) {
            auto tmp = knn[i];
            float num = 0;
            if (k == 1)
                std::cout << "tmp: " << tmp.top().second;
            while (!tmp.empty()) {
                for (size_t j = 0; j < topK; j++) {
                    if (gt[i * k + j] == tmp.top().second) {
                        num++;
                        break;
                    }
                }
                tmp.pop();
            }
            res[num]++;
            recall += num / topK;
        }

        for (int i = 0; i <= 10; i++)
            std::cout << res[i] << std::endl;
        std::cout << recall / nq << std::endl;
    }
};

#endif  // INDEX_H
