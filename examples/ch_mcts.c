#include "darknet.h"
#include "ch_mcts.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <stdint.h>
#include <sys/time.h>
#include "system.h"
#include <stdio.h>

static void mcts_fatal(const char* what) {
    fprintf(stderr, "[MCTS] FATAL: %s\n", what ? what : "(unknown)");
}

static double now_sec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

static inline unsigned seed_from_ptr(const void* p) {
    return (unsigned)((uintptr_t)p ^ (uintptr_t)time(NULL));
}

static inline void fisher_yates_shuffle(int* idx, int n, unsigned* seed) {
    for (int i = n - 1; i > 0; --i) {
        int j = rand_r(seed) % (i + 1);
        int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
}

static void* arena_alloc(ch_mcts* m, size_t bytes, size_t align) {
    size_t base = (size_t)m->arena + m->arena_used;
    size_t aligned = (base + (align - 1)) & ~(align - 1);
    size_t new_used = (aligned - (size_t)m->arena) + bytes;
    if (new_used > m->arena_size) return NULL;
    m->arena_used = new_used;
    return (void*)aligned;
}

static ch_mcts_node* alloc_node(ch_mcts* m) {
    ch_mcts_node* n = (ch_mcts_node*)arena_alloc(m, sizeof(ch_mcts_node), 8);
    if (!n) return NULL;
    memset(n, 0, sizeof(*n));
    n->children    = (ch_mcts_node**)arena_alloc(m, sizeof(ch_mcts_node*) * m->cfg.max_children, 8);
    n->moves       = (ch_mv*)arena_alloc(m, sizeof(ch_mv) * m->cfg.max_children, 8);
    n->priors      = (float*)arena_alloc(m, sizeof(float) * m->cfg.max_children, 8);
    n->value_sum   = (float*)arena_alloc(m, sizeof(float) * m->cfg.max_children, 8);
    n->visit_count = (int*)arena_alloc(m, sizeof(int) * m->cfg.max_children, 8);
    n->state       = (ch_board*)arena_alloc(m, sizeof(ch_board), 8);
    if (!n->children || !n->moves || !n->priors || !n->value_sum || !n->visit_count || !n->state) return NULL;
    return n;
}

int ch_mcts_init(ch_mcts* m, ch_api api, ch_mcts_config cfg, ch_nets* nets, void* arena, size_t arena_size) {
    if (!m || !arena || arena_size == 0) return -1;
    m->api = api;
    m->cfg = cfg;
    m->nets = nets;
    m->arena = arena;
    m->arena_size = arena_size;
    m->arena_used = 0;
    return 0;
}

ch_mcts_node* ch_mcts_create_root(ch_mcts* m, const ch_board* start, int to_move) {
    ch_mcts_node* r = alloc_node(m);
    if (!r) return NULL;
    memcpy(r->state, start, sizeof(ch_board));
    r->parent = NULL;
    r->to_move = to_move;
    r->is_expanded = 0;
    r->is_terminal = 0;
    r->total_visits = 0;
    return r;
}

float ch_mcts_child_q(const ch_mcts_node* node, int i) {
    int n = node->visit_count[i];
    if (n <= 0) return 0.0f;
    return node->value_sum[i] / (float)n;
}

float ch_mcts_child_u(const ch_mcts* m, const ch_mcts_node* node, int i) {
    float prior = node->priors[i];
    float parent_visits = (float)fmax(1, node->total_visits);
    float denom = 1.0f + (float)node->visit_count[i];
    float u = m->cfg.cpuct * prior * sqrtf(parent_visits) / denom;
    if (!isfinite(u)) u = 0.0f;
    return u;
}

ch_mcts_node* select_leaf(ch_mcts* m, ch_mcts_node* node) {
    if (!node) { mcts_fatal("select_leaf: node is NULL"); return NULL; }
    while (node->is_expanded && !node->is_terminal && node->child_count > 0) {
        int N = node->child_count;
        if (N <= 0) break;

        int best = -1;
        float best_score = -FLT_MAX;

        int* order = (int*)alloca(sizeof(int)*N);
        for (int i = 0; i < N; ++i) order[i] = i;
        unsigned seed = seed_from_ptr(node);
        fisher_yates_shuffle(order, N, &seed);

        for (int t = 0; t < N; ++t) {
            int i = order[t];
            float q = ch_mcts_child_q(node, i);
            float u = ch_mcts_child_u(m, node, i);
            float score = q + u;
            if (!isfinite(score)) continue;
            if (best == -1 || score > best_score) { best = i; best_score = score; }
        }

        if (best == -1) best = 0;                  // all scores invalid -> fallback
        if (best < 0 || best >= N) best = 0;       // hard clamp

        ch_mcts_node* child = node->children[best];
        if (!child) { mcts_fatal("select_leaf: child pointer is NULL"); return node; }

        if (m->cfg.use_virtual_loss) {
            node->visit_count[best] += 1;
            node->value_sum[best]   -= m->cfg.virtual_loss;
            node->total_visits      += 1;
        }
        node = child;
    }
    return node;
}

static inline void ch_mcts_set_node_value(ch_mcts_node *n, float v) { n->node_value = v; }

static int expand_and_eval(ch_mcts* m, ch_mcts_node* node, float* out_value) {
    int term = m->api.terminal_result(node->state, node->to_move);
    if (term != 0) {
        node->is_terminal = 1; node->is_expanded = 1; node->child_count = 0;
        *out_value = (float)term; ch_mcts_set_node_value(node, *out_value);
        return 0;
    }

    ch_mv* tmp_moves = (ch_mv*)alloca(sizeof(ch_mv) * m->cfg.max_children);
    int genN = m->api.gen_legal(node->state, tmp_moves, m->cfg.max_children);
    if (genN <= 0) {
        node->is_terminal = 1; node->is_expanded = 1; node->child_count = 0;
        *out_value = 0.0f; ch_mcts_set_node_value(node, 0.0f);
        return 0;
    }

    float* pri = (float*)alloca(sizeof(float) * CH_ACTION_SPACE);
    float val = 0.0f;
    if (m->api.nn_eval(m->nets, node->state, tmp_moves, genN, pri, &val) != 0) {
        mcts_fatal("expand_and_eval: nn_eval failed");
        return -1;
    }


    if (node->to_move) { val = -val; }
// Normalize priors robustly
    double s = 0.0;
    for (int i = 0; i < genN; ++i) {
        if (!isfinite(pri[i]) || pri[i] < 0) pri[i] = 0.0f;
        s += pri[i];
    }
    if (!(s > 0.0)) { // also handles NaN
        float uni = 1.0f / (float)genN;
        for (int i = 0; i < genN; ++i) pri[i] = uni;
    } else {
        float inv = 1.0f / (float)s;
        for (int i = 0; i < genN; ++i) pri[i] *= inv;
    }

    // Root noise (Dirichlet)
    if (!node->parent && m->cfg.dirichlet_alpha > 0 && m->cfg.root_noise_frac > 0) {
        double* g = (double*)alloca(sizeof(double)*genN);
        unsigned seed = seed_from_ptr(node);
        double sumg = 0.0;
        int k = (int)fmax(1.0, m->cfg.dirichlet_alpha * 10.0);
        for (int i = 0; i < genN; ++i) {
            double acc = 0.0;
            for (int j = 0; j < k; ++j) {
                double u = (rand_r(&seed)+1.0) / (RAND_MAX+2.0);
                acc += -log(u);
            }
            g[i] = acc; sumg += g[i];
        }
        if (sumg > 0) {
            for (int i = 0; i < genN; ++i) g[i] /= sumg;
            float eps = m->cfg.root_noise_frac;
            for (int i = 0; i < genN; ++i) pri[i] = (1.0f - eps) * pri[i] + eps * (float)g[i];
        }
    }

    // Create children
    node->child_count = genN;
    for (int i = 0; i < genN; ++i) {
        node->moves[i] = tmp_moves[i];
        node->priors[i] = pri[i];
        node->value_sum[i] = 0.0f;
        node->visit_count[i] = 0;

        ch_mcts_node* ch = alloc_node(m);
        if (!ch) { mcts_fatal("expand_and_eval: alloc_node child failed"); return -1; }
        node->children[i] = ch;
        ch->parent = node;
        ch->move_from_parent = node->moves[i];
        ch->to_move = -node->to_move;
        if (m->api.apply_move(node->state, node->moves[i], ch->state) != 0) {
            mcts_fatal("expand_and_eval: apply_move failed");
            return -1;
        }
    }

    node->is_expanded = 1;
    node->is_terminal = 0;
    *out_value = val;
    ch_mcts_set_node_value(node, val);
    return 0;
}

static void backpropagate(ch_mcts* m, ch_mcts_node* leaf, float value, int used_virtual_loss) {
    (void)m;
    ch_mcts_node* node = leaf;
    int first = 1;
    while (node && node->parent) {
        ch_mcts_node* parent = node->parent;
        int idx = -1;
        for (int i = 0; i < parent->child_count; ++i) {
            if (parent->children[i] == node) { idx = i; break; }
        }
        if (idx < 0) { mcts_fatal("backpropagate: child index not found"); return; }

        if (used_virtual_loss && first && m->cfg.use_virtual_loss) {
            parent->visit_count[idx] -= 1;
            parent->value_sum[idx]   += m->cfg.virtual_loss;
            parent->total_visits     -= 1;
        }
        parent->visit_count[idx] += 1;
        parent->value_sum[idx]   += value;
        parent->total_visits     += 1;

        value = -value;
        node = parent;
        first = 0;
    }
}

static void run_simulation(ch_mcts* m, ch_mcts_node* root) {
    if (!root) { mcts_fatal("run_simulation: root is NULL"); return; }
    ch_mcts_node* leaf = select_leaf(m, root);
    if (!leaf) { mcts_fatal("run_simulation: leaf is NULL"); return; }

    float value = 0.0f;
    int rc = 0;
    if (!leaf->is_expanded) {
        rc = expand_and_eval(m, leaf, &value);
        if (rc != 0) return;
    } else {
        int term = m->api.terminal_result(leaf->state, leaf->to_move);
        value = (float)term;
    }
    backpropagate(m, leaf, value, 1);
}

void ch_mcts_run(ch_mcts* m, ch_mcts_node* root, int seconds) {
    if (!root) { mcts_fatal("ch_mcts_run: root is NULL"); return; }

    double start = now_sec();
    int total_sim = 0;

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic,1) reduction(+:total_sim)
        for (int t = 0; t < 1000000000; ++t) {
            double elapsed = now_sec() - start;
            if (elapsed >= seconds) break;
            run_simulation(m, root);
            total_sim++;
        }
    }
}


static int sample_from_visits(const ch_mcts_node* root, float tau, int* out_idx) {
    int N = root->child_count; if (N <= 0) return -1;
    if (tau <= 1e-6f) {
        int best = 0, bestv = root->visit_count[0];
        for (int i = 1; i < N; ++i) { if (root->visit_count[i] > bestv) { bestv = root->visit_count[i]; best = i; } }
        *out_idx = best; return 0;
    }
    double sum = 0.0; double* w = (double*)alloca(sizeof(double)*N); double invtau = 1.0 / (double)tau;
    for (int i = 0; i < N; ++i) {
        double p = pow(fmax(1.0, (double)root->visit_count[i]), invtau);
        w[i] = p; sum += p;
    }
    if (sum <= 0) return -1;
    double r = ((double)rand() / (double)RAND_MAX) * sum; double acc = 0.0;
    for (int i = 0; i < N; ++i) { acc += w[i]; if (r <= acc) { *out_idx = i; return 0; } }
    *out_idx = N - 1; return 0;
}

ch_mv ch_mcts_pick_move(const ch_mcts* m, const ch_mcts_node* root, float tau) {
    (void)m;
    int idx = 0;
    if (sample_from_visits(root, tau, &idx) != 0) return (ch_mv)0;
    return root->moves[idx];
}

void ch_mcts_free_all(ch_mcts* m) { (void)m; }

int ch_mcts_root_visits(const ch_mcts_node* root, int* out_counts, int max) {
    int n = root->child_count;
    if (max < n) n = max;
    for (int i = 0; i < n; ++i) out_counts[i] = root->visit_count[i];
    return n;
}

static void print_top_children(const ch_mcts_node* root, char** moves, int topN) {
    int N = root->child_count;
    if (N <= 0) { fprintf(stderr, "(no children)\n"); return; }
    if (topN > N) topN = N;

    int *idx = (int*)alloca(sizeof(int)*topN);
    float *score = (float*)alloca(sizeof(float)*topN);
    for (int i = 0; i < topN; ++i) { idx[i] = i; score[i] = (float)root->visit_count[i]; }

    for (int i = 0; i < topN; ++i) {
        for (int j = i + 1; j < topN; ++j) {
            if (score[j] > score[i]) {
                float ts = score[i]; score[i] = score[j]; score[j] = ts;
                int ti = idx[i]; idx[i] = idx[j]; idx[j] = ti;
            }
        }
    }
    fprintf(stderr, "top %d candidates:\n", topN);
    for (int k = 0; k < topN; ++k) {
        int i = idx[k];
        float q = ch_mcts_child_q(root, i);
        if (moves && (int)root->moves[i] >= 0 && (int)root->moves[i] < root->child_count) {
        fprintf(stderr, "#%2d move: %s visits: %d Q: %.3f prior: %.3f\n",
                k + 1, moves[(int)root->moves[i]], root->visit_count[i], q, root->priors[i]);
        } else {
            fprintf(stderr, "#%2d move: %llu visits: %d Q: %.3f prior: %.3f\n",
                    k + 1, (unsigned long long)root->moves[i], root->visit_count[i], q, root->priors[i]);
        }
    }
}

void ch_mcts_play_and_log(ch_mcts* m, ch_mcts_node* root, int seconds, char** moves) {
    if (!m || !root) return;

    double start = now_sec();
    int total_sim = 0;

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic,1) reduction(+:total_sim)
        for (int t = 0; t < 1000000000; ++t) {
            double elapsed = now_sec() - start;
            if (elapsed >= seconds) break;

            ch_mcts_node* leaf = select_leaf(m, root);
            if (!leaf) continue;

            float value = 0.f;
            if (expand_and_eval(m, leaf, &value) != 0) continue;
            backpropagate(m, leaf, value, m->cfg.use_virtual_loss);

            total_sim++;
        }
    }

    double elapsed = now_sec() - start;
    ch_mv best_move = ch_mcts_pick_move(m, root, 1.0f);

    const char* best_str = NULL;
    if (moves && (int)best_move >= 0 && (int)best_move < root->child_count) {
        best_str = moves[(int)best_move];
    } else {
        static char tmp[32];
        snprintf(tmp, sizeof(tmp), "%llu", (unsigned long long)best_move);
        best_str = tmp;
    }

    if (root) {
        ch_mcts_diag(m, root, elapsed, best_move);
        fprintf(stderr, "chosen move: %s\n", best_str);
        print_top_children(root, moves, 8);
    }
}

int ch_mcts_export_root(const ch_mcts_node* root, int* out_moves, int* out_visits, float* out_q, float* out_priors, int max) {
    if (!root) return -1;
    int n = root->child_count;
    if (max > 0 && n > max) n = max;
    for (int i = 0; i < n; ++i) {
        if (out_moves)  out_moves[i]  = (int)root->moves[i];
        if (out_visits) out_visits[i] = root->visit_count[i];
        if (out_q)      out_q[i]      = ch_mcts_child_q(root, i);
        if (out_priors) out_priors[i] = root->priors[i];
    }
    return n;
}

int ch_mcts_policy_over_children(const ch_mcts_node* root, float tau, float* out_pi, int max) {
    if (!root || !out_pi) return -1;
    int n = root->child_count;
    if (max > 0 && n > max) n = max;
    if (n <= 0) return 0;
    double sum = 0.0;
    double invtau = (tau <= 1e-6f) ? 0.0 : 1.0 / (double)tau;
    for (int i = 0; i < n; ++i) {
        double p;
        if (tau <= 1e-6f) {
            p = (i == 0) ? 1.0 : 0.0;
        } else {
            p = pow(fmax(1.0, (double)root->visit_count[i]), invtau);
        }
        out_pi[i] = (float)p;
        sum += p;
    }
    if (sum <= 0.0) {
        float u = 1.0f / (float)n;
        for (int i = 0; i < n; ++i) out_pi[i] = u;
    } else {
        float inv = 1.0f / (float)sum;
        for (int i = 0; i < n; ++i) out_pi[i] = out_pi[i] * inv;
    }
    return n;
}

void ch_mcts_diag(ch_mcts* m, ch_mcts_node* root, double elapsed, ch_mv best_move) {
    float sumQ = 0.f, maxQ = -FLT_MAX;
    for (int i = 0; i < root->child_count; ++i) {
        float q = ch_mcts_child_q(root, i);
        sumQ += q;
        if (q > maxQ) maxQ = q;
    }
    float avgQ = (root->child_count > 0) ? sumQ / root->child_count : 0.f;
    fprintf(stderr,
        "mcts time: %.2fs total_visits: %d avg_Q: %.3f max_Q: %.3f cpuct: %.3f noise: %.3f virt_loss: %.3f\n",
        elapsed, root->total_visits, avgQ, maxQ,
        m->cfg.cpuct, m->cfg.root_noise_frac, m->cfg.virtual_loss
    );
}