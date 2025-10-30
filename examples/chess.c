#include "darknet.h"
#include "ch_mcts.h"
#include "system.h"
#include "image.h"
#include "time.h"
#include "opencl.h"
#include <string.h>

#include <stdlib.h>

#include <stdio.h>
#ifdef WIN32
#include "unistd\dirent.h"
#else
#include <dirent.h>
#endif

#ifdef WIN32
#include "unistd\unistd.h"
#else
#include <unistd.h>
#endif

#include <sys/stat.h>
#include <assert.h>

#define class temp

#include <execinfo.h>

#define FEN_MAX 256
#define MOVE_MAX 64

void ASSERT(int condition) {
    if (!condition) {
        void *buffer[1000];
        int size = backtrace(buffer, 1000);
        char **symbols = backtrace_symbols(buffer, size);
        if (symbols) {
            fprintf(stderr, "call Stack:\n");
            for (int i = 0; i < size; ++i) {
                fprintf(stderr, "%s\n", symbols[i]);
            }
            FREE(symbols);
        } else {
            fprintf(stderr, "failed to retrieve call stack symbols!\n");
        }
    }
}

static ch_nets *g_nets = NULL;

static ch_nets *ch_load_networks(char *cfg, char *weights) {
    ch_nets *nets = (ch_nets*)CALLOC(1, sizeof(ch_nets));

    char *v_cfg = (cfg && cfg[0]) ? cfg : "chess.cfg";
    char *v_weights = (weights && weights[0]) ? weights : "chess.weights";

    char *p_cfg = NULL;
    char *p_weights = NULL;

    if (strstr(v_cfg, ".cfg")) {
        p_cfg = strdup(v_cfg);
        char *dot = strrchr(p_cfg, '.');
        if (dot) strcpy(dot, "p.cfg");
    } else {
        size_t len = strlen(v_cfg) + 3;
        p_cfg = (char*)CALLOC(len, 1);
        snprintf(p_cfg, len, "%sp.cfg", v_cfg);
    }

    if (strstr(v_weights, ".weights")) {
        p_weights = strdup(v_weights);
        char *dot = strrchr(p_weights, '.');
        if (dot) strcpy(dot, "p.weights");
    } else {
        size_t len = strlen(v_weights) + 10;
        p_weights = (char*)CALLOC(len, 1);
        snprintf(p_weights, len, "%sp.weights", v_weights);
    }

    nets->netv_name = strdup(v_cfg);
    nets->netv_wname = strdup(v_weights);
    nets->netp_name = strdup(p_cfg);
    nets->netp_wname = strdup(p_weights);

    if (!exists(v_weights)) {
        fprintf(stderr, "Parsing network:  %s\n", v_cfg);
        nets->netv = parse_network_cfg(v_cfg);
    }
    else {
        fprintf(stderr, "Loading value network:  %s (%s)\n", v_cfg, v_weights);
        nets->netv = load_network(v_cfg, v_weights, 0);
        if (!nets->netv) {
            fprintf(stderr, "Failed to load value net\n");
            FREE(nets);
            FREE(p_cfg);
            FREE(p_weights);
            return NULL;
        }
    }

    if (!exists(p_weights)) {
        fprintf(stderr, "Parsing network:  %s\n", p_cfg);
        nets->netp = parse_network_cfg(p_cfg);
    }
    else {
        fprintf(stderr, "Loading policy network: %s (%s)\n", p_cfg, p_weights);
        nets->netp = load_network(p_cfg, p_weights, 0);
        if (!nets->netp) {
            fprintf(stderr, "Failed to load policy net\n");
            free_network(nets->netv);
            FREE(nets);
            FREE(p_cfg);
            FREE(p_weights);
            return NULL;
        }
    }

    fprintf(stderr, "Loaded networks OK\n");

    g_nets = nets;

    FREE(p_cfg);
    FREE(p_weights);
    return nets;
}

static void ch_free_networks(ch_nets *nets) {
    if (!nets) return;
    if (nets->netv) free_network(nets->netv);
    if (nets->netp) free_network(nets->netp);
    if (nets->netv_name) FREE(nets->netv_name);
    if (nets->netp_name) FREE(nets->netp_name);
    FREE(nets);
    if (g_nets == nets) g_nets = NULL;
}

static int ch_eval_dual(ch_nets* nets, const ch_board* board, const ch_mv* moves, int moves_count, float* out_policy, float* out_value)
{
    if (!nets || !board || !out_policy || !out_value) return -1;

    float *input = ch_fen_to_board((char*)board->fen, 1);
    if (!input) return -1;

    nets->netv->train = 0;
    memcpy(nets->netv->input, input, nets->netv->inputs * sizeof(float));
    forward_network(nets->netv);

    float v = nets->netv->output[0];
    if (!isfinite(v)) v = 0.f;
    if (v < -1.f) v = -1.f;
    if (v >  1.f) v =  1.f;
    *out_value = v;

    nets->netp->train = 0;
    memcpy(nets->netp->input, input, nets->netp->inputs * sizeof(float));
    forward_network(nets->netp);

    int n = moves_count > 0 ? moves_count : 1;
    float uniform = 1.0f / (float)n;
    for (int i = 0; i < moves_count; ++i) {
        float p = (i < nets->netp->outputs) ? nets->netp->output[i] : 0.f;
        if (!isfinite(p) || p < 0.f) p = 0.f;
        out_policy[i] = 0.5f * p + 0.5f * uniform;
    }

    float s = 1e-8f;
    for (int i = 0; i < moves_count; ++i) s += out_policy[i];
    float inv = 1.f / s;
    for (int i = 0; i < moves_count; ++i) out_policy[i] *= inv;

    FREE(input);
    return 0;
}

static float ch_train_dual(ch_nets* nets, const float *x, const float *pi_full, float v_target, float capture_bonus) {
    if (!nets || !x || !pi_full) return 0.f;

    nets->netp->seen += nets->netp->batch;
    nets->netp->train = 1;
    memcpy(nets->netp->input, x, nets->netp->inputs * sizeof(float));
    forward_network(nets->netp);

    float sum = 1e-8f;
    for (int i = 0; i < CH_ACTION_SPACE; ++i)
        if (isfinite(pi_full[i]) && pi_full[i] > 0.f)
            sum += pi_full[i];

    // dynamic uniform-policy blend (starts 0.5 → min 0.2)
    float mix = fmaxf(0.2f, 0.5f - 0.000001f * (float)nets->netp->nsteps);

    float entropy = 0.f;
    for (int i = 0; i < CH_ACTION_SPACE; ++i) {
        float pi = fmaxf(0.f, pi_full[i]) / sum;
        pi = (1.0f - mix) * pi + mix / CH_ACTION_SPACE;
        nets->netp->truth[i] = pi;
        if (pi > 1e-6f) entropy -= pi * logf(pi);
    }

    backward_network(nets->netp);
    float errorp = *nets->netp->cost;
    update_network(nets->netp);

    nets->netv->seen += nets->netv->batch;
    nets->netv->train = 1;
    memcpy(nets->netv->input, x, nets->netv->inputs * sizeof(float));
    forward_network(nets->netv);

    float v_pred = nets->netv->output[0];
    if (!isfinite(v_pred)) v_pred = 0.f;
    if (!isfinite(v_target)) v_target = 0.f;
    v_pred = fminf(fmaxf(v_pred, -1.f), 1.f);
    v_target = fminf(fmaxf(v_target, -1.f), 1.f);

    // tactical shaping — capture bonus
    float v_combined = 0.8f * (v_target + capture_bonus) + 0.2f * v_pred;

    // amplify strong wins slightly
    if (fabsf(v_target) > 0.9f)
        v_combined *= 1.1f;

    nets->netv->truth[0] = v_combined;

    backward_network(nets->netv);
    float errorv = *nets->netv->cost;
    update_network(nets->netv);

    const float value_weight = 1.0f;
    const float policy_weight = 0.01f;
    const float entropy_weight = 0.001f * expf(-0.00005f * (float)nets->netv->nsteps);

    float loss = value_weight * errorv + policy_weight * errorp - entropy_weight * entropy;

    size_t iter = nets->netp->nsteps;
    fprintf(stderr,
        "train step: %lu batch: %d v_pred: %.3f v_tgt: %.3f cap_bonus: %.3f | rate: %.4f loss: %.4f (V: %.4f P: %.4f E: %.4f mix: %.3f)\n",
        iter,
        nets->netp->batch,
        v_pred, v_target, capture_bonus,
        nets->netp->learning_rate,
        loss,
        errorv, errorp, entropy, mix
    );

    return loss;
}

static int ch_test_tchess_count = -1;

typedef struct {
    char* fen;
    char **moves;
    int n;
} ch_moves;

typedef struct ch_constant_memory_queue {
    void** data;
    void *tree;
    int capacity;
    int count;
    int put_id;
    int get_id;
    int peak_id;
    int total_count;
    int index;
    float value;
    float power;
} ch_constant_memory_queue;

ch_constant_memory_queue* ch_create_constant_memory_queue(int capacity) {
    ch_constant_memory_queue* queue = (ch_constant_memory_queue*) CALLOC(1, sizeof(ch_constant_memory_queue));
    queue->data = (void**) CALLOC(capacity, sizeof(void*));
    queue->capacity = capacity;
    for(int i = 0; i < queue->capacity; ++i) queue->data[i] = NULL;
    queue->count = 0;
    queue->put_id = -1;
    queue->get_id = -1;
    queue->peak_id = -1;
    queue->index = -1;
    return queue;
}

void ch_clean_constant_memory_queue(ch_constant_memory_queue* queue){
    for(int i = 0; i < queue->count; ++i) if (queue->data[i] != NULL) { FREE(queue->data[i]); queue->data[i] = NULL; }
    queue->count = 0;
    queue->peak_id = -1;
    queue->get_id = -1;
    queue->put_id = -1;
}

int ch_get_next_put_id(ch_constant_memory_queue* queue) {
    int put_id = queue->capacity > queue->put_id + 1 ? ++queue->put_id : 0;
    return put_id;
}

void ch_enqueue(ch_constant_memory_queue* queue, void* item) {
    int put_id = ch_get_next_put_id(queue);
    queue->data[put_id] = item;
    queue->count = queue->count < queue->capacity ? queue->count + 1 : queue->capacity;
}

int ch_get_prev_get_id(ch_constant_memory_queue* queue) {
    int get_id = 0 < queue->get_id - 1 ? --queue->get_id : queue->capacity;
    return get_id;
}

int ch_get_next_get_id(ch_constant_memory_queue* queue) {
    int get_id = queue->capacity > queue->get_id + 1 ? ++queue->get_id : 0;
    return get_id;
}

void* ch_rollback(ch_constant_memory_queue* queue) {
    int get_id = ch_get_prev_get_id(queue);
    queue->count = queue->count > 0 ? queue->count - 1 : 0;
    return queue->data[get_id];
}

void* ch_dequeue(ch_constant_memory_queue* queue) {
    int get_id = ch_get_next_get_id(queue);
    queue->count = queue->count > 0 ? queue->count - 1 : 0;
    return queue->data[get_id];
}

int ch_queue_count(ch_constant_memory_queue* queue) {
    if (queue == NULL) return 0;
    return queue->count;
}

int ch_is_empty(ch_constant_memory_queue* queue) {
    if (queue == NULL) return 0;
    return queue->count == 0;
}

void ch_constant_memory_queue_peek_init(ch_constant_memory_queue* queue) {
    queue->peak_id = queue->get_id;
}

void* ch_constant_memory_queue_peek(ch_constant_memory_queue* queue) {
    int peak_id = queue->capacity > queue->peak_id + 1 ? ++queue->peak_id : 0;
    void* item = queue->data[peak_id];
    return item;
}

void ch_constant_memory_queue_replace(ch_constant_memory_queue* queue, void* item) {
    int peak_id = queue->put_id;
    queue->data[peak_id] = item;
}

void ch_destroy_constant_memory_queue(ch_constant_memory_queue* queue) {
    if (queue == NULL) return;
    for(int i = 0; i < queue->count; ++i) if (queue->data[i] != NULL) { FREE(queue->data[i]); queue->data[i] = NULL; }
    queue->capacity = 0;
    queue->count = 0;
    queue->peak_id = -1;
    queue->get_id = -1;
    queue->put_id = -1;
    FREE(queue->data);
    queue->data = NULL;queue->tree = NULL;
    FREE(queue);
    queue = NULL;
}

char *start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

static float* empty_board = NULL;
static float* start_board = NULL;

typedef struct ch_dict_t {
    char *key;
    void *value;
    struct ch_dict_t *next;
    time_t stamp;
} ch_dict;

ch_dict **ch_dict_alloc() {
    return CALLOC(1, sizeof(ch_dict));
}

void ch_dict_dealloc(ch_dict **dict) {
    FREE(dict);
}

void* ch_dict_get(ch_dict *dict, char *key) {
    ch_dict *ptr;
    for (ptr = dict; ptr != NULL; ptr = ptr->next) {
        if (strcmp(ptr->key, key) == 0) {
            return ptr->value;
        }
    }
    return NULL;
}

void ch_dict_del(ch_dict **dict, char *key) {
    ch_dict *ptr, *prev;
    for (ptr = *dict, prev = NULL; ptr != NULL; prev = ptr, ptr = ptr->next) {
        if (strcmp(ptr->key, key) == 0) {
            if (ptr->next != NULL) {
                if (prev == NULL) {
                    *dict = ptr->next;
                } else {
                    prev->next = ptr->next;
                }
            } else if (prev != NULL) {
                prev->next = NULL;
            } else {
                *dict = NULL;
            }
            FREE(ptr->key);
            ptr->key = NULL;
            /*
            if (ptr->value != NULL) {
                FREE(ptr->value);
                ptr->value = NULL;
            }
            */
            FREE(ptr);
            ptr = NULL;
            return;
        }
    }
}

void ch_dict_exp(ch_dict **dict) {
    ch_dict *ptr, *prev;
    for (ptr = *dict, prev = NULL; ptr != NULL; prev = ptr, ptr = ptr->next) {
        double datetime_diff_ms = difftime(time(0), ptr->stamp) * 1000.;
        if (datetime_diff_ms > (60. * 60. * 1000.)) {
            if (ptr->next != NULL) {
                if (prev == NULL) {
                    *dict = ptr->next;
                } else {
                    prev->next = ptr->next;
                }
            } else if (prev != NULL) {
                prev->next = NULL;
            } else {
                *dict = NULL;
            }
            FREE(ptr->key);
            ptr->key = NULL;
            if (ptr->value != NULL) {
                FREE(ptr->value);
                ptr->value = NULL;
            }
            FREE(ptr);
            ptr = NULL;
            return;
        }
    }
}

void ch_dict_add(ch_dict **dict, char *key, void *value) {
    ch_dict_exp(dict);
    ch_dict *gd = ch_dict_get(*dict, key);
    if (gd != NULL)
    {
        gd->stamp = time(0);
        *dict = gd;
        return;
    }
    ch_dict *d = (ch_dict *) CALLOC(1, sizeof(ch_dict));
    d->key = (char*) CALLOC(strlen(key)+1, sizeof(char));
    strcpy(d->key, key);
    d->value = value;
    d->next = *dict;
    d->stamp = time(0);
    *dict = d;
}

static ch_dict* moves_history = NULL;
static ch_dict* moves_history_positions = NULL;

#define BOARD_SIZE (8*8+8)

typedef struct ch_learn_state_t {
    float board[BOARD_SIZE];
    int index;
    int player;
} ch_learn_state;

static ch_learn_state* empty_item = NULL;

void ch_init_game_history(char* sessionId) {
    if (empty_board == NULL) {
        empty_board = (float*) CALLOC(BOARD_SIZE, sizeof(float));
        for (int i = 0; i < BOARD_SIZE; ++i) {
            empty_board[i] = 0;
        }
    }
    if (start_board == NULL) {
        start_board = ch_fen_to_board(start_fen, 0);
    }
    if (empty_item == NULL) {
        empty_item = (ch_learn_state*) CALLOC(1, sizeof(ch_learn_state));
        empty_item->index = -1;
        memcpy(empty_item->board, empty_board, (1)*(BOARD_SIZE)*sizeof(float));
        empty_item->player = -1;
    }
    if (moves_history == NULL) {
        moves_history = *ch_dict_alloc();
    }
    if (ch_dict_get(moves_history, sessionId) == NULL) {
        ch_dict_add(&moves_history, sessionId, ch_create_constant_memory_queue(512));
    }
    if (moves_history_positions == NULL) {
        moves_history_positions = *ch_dict_alloc();
    }
    if (ch_dict_get(moves_history_positions, sessionId) == NULL) {
        ch_dict_add(&moves_history_positions, sessionId, ch_create_constant_memory_queue(512));
    }
}

void ch_put_back(char* sessionId, ch_learn_state* item) {
    ch_init_game_history(sessionId);
    ch_constant_memory_queue *q = ch_dict_get(moves_history, sessionId);
    ch_enqueue(q, item);
}

int trivial_player = -1;

void ch_clean_history(char* sessionId, int init) {
    ch_constant_memory_queue *q1 = ch_dict_get(moves_history, sessionId);
    ch_destroy_constant_memory_queue(q1);
    ch_dict_del(&moves_history, sessionId);
    ch_constant_memory_queue *q2 = ch_dict_get(moves_history_positions, sessionId);
    ch_destroy_constant_memory_queue(q2);
    ch_dict_del(&moves_history_positions, sessionId);
    if (init) {
        ch_learn_state *to_learn = (ch_learn_state *) CALLOC(1, sizeof(ch_learn_state));
        memcpy(to_learn->board, start_board, (1)*(BOARD_SIZE)*sizeof(float));
        to_learn->index = 0;
        to_learn->player = 0;
        ch_put_back(sessionId, to_learn);
    }
    if (trivial_player != -1) trivial_player = trivial_player == 0 ? 1 : 0;
}

ch_learn_state* ch_pick(char* sessionId) {
    ch_init_game_history(sessionId);
    ch_constant_memory_queue *q = ch_dict_get(moves_history, sessionId);
    ch_learn_state* queued = (ch_learn_state*)ch_rollback(q);
    return queued;
}

ch_learn_state* ch_pick_back(char* sessionId) {
    ch_init_game_history(sessionId);
    ch_constant_memory_queue *q = ch_dict_get(moves_history, sessionId);
    ch_learn_state* queued = (ch_learn_state*)ch_dequeue(q);
    return queued;
}

ch_moves ch_load_moves(char* valid_fen, char** valid_moves, int valid_moves_count)
{
    ch_moves m = {0};
    m.fen = valid_fen;
    m.moves = valid_moves;
    m.n = valid_moves_count;
    return m;
}

typedef struct ch_position {
    char fen_board[128];
    int move_count;
} ch_position;

char* ch_parse_fen(char* fen) {
    int i = 0;
    while (fen[i] != ' ' && fen[i] != '\0') {
        i++;
    }
    char* fen_board = (char*) CALLOC(i+1, sizeof(char));
    strncpy(fen_board, fen, i);
    fen_board[i] = '\0';
    return fen_board;
}

float* ch_fen_to_board_with_history(char* sessionId, char* valid_fen, int n) {
    if (moves_history == NULL) {
        moves_history = *ch_dict_alloc();
    }
    if(ch_dict_get(moves_history, sessionId) == NULL) {
        ch_dict_add(&moves_history, sessionId, ch_create_constant_memory_queue(512));
    }

    float* value = (float*) CALLOC(n*(BOARD_SIZE), sizeof(float));

    ch_constant_memory_queue *q = ch_dict_get(moves_history, sessionId);
    int count = ch_queue_count(q);
    ch_constant_memory_queue_peek_init(q);

    float* valid_board = ch_fen_to_board(valid_fen, 1);

    for(int i = 0; i < n; ++i) {
        if (count < i) {
            ch_learn_state *current_position = ch_constant_memory_queue_peek(q);
            for (int k = 0; k < 8 * 8; ++k) value[i * 8 * 8 + k] = current_position->board[k];
        } else {
            for (int j = i; j < n; ++j) {
                for (int k = 0; k < 8 * 8; ++k) value[i * 8 * 8 + k] = valid_board[k];
            }
            break;
        }
    }

    FREE(valid_board);

    return value;
}

int ch_is_three_fold_repetition(char* sessionId, char* fen) {
    if (moves_history_positions == NULL) {
        moves_history_positions = *ch_dict_alloc();
    }

    if(ch_dict_get(moves_history_positions, sessionId) == NULL) {
        ch_dict_add(&moves_history_positions, sessionId, ch_create_constant_memory_queue(512));
    }

    if (fen == NULL || fen[0] == '\0') {
        ch_constant_memory_queue *q = ch_dict_get(moves_history_positions, sessionId);
        ch_destroy_constant_memory_queue(q);
        ch_dict_del(&moves_history_positions, sessionId);
        q = NULL;
        return 0;
    }

    ch_constant_memory_queue *q = ch_dict_get(moves_history_positions, sessionId);

    if (q == NULL) {
        q = ch_create_constant_memory_queue(512);
        ch_dict_add(&moves_history_positions, sessionId, q);
    }

    int count = ch_queue_count(q);

    if (count > 150) {
        count = ch_queue_count(q);
        for (int i = 0; i < count; ++i) {
            ch_position *peek_position = ch_dequeue(q);
        }
        return 1;
    }

    ch_constant_memory_queue_peek_init(q);

    char* position_fen = ch_parse_fen(fen);

    int fault = 0;
    for (int i = 0; i < count; ++i) {
        ch_position *current_position = ch_constant_memory_queue_peek(q);
        if (current_position != NULL && strlen(current_position->fen_board) != 0) {
            if (strcmp(position_fen, current_position->fen_board) == 0) {
                current_position->move_count += 1;
                if (current_position->move_count >= 3) {
                    fault = 1;
                    break;
                }
            }
        }
    }

    if (fault == 1) {
        count = ch_queue_count(q);
        for (int i = 0; i < count; ++i) {
            ch_position *peek_position = ch_dequeue(q);
        }
        FREE(position_fen);
        return fault;
    }

    ch_position* position = (ch_position*) CALLOC(1, sizeof(ch_position));
    strcpy(position->fen_board, position_fen);
    position->move_count = 1;
    ch_enqueue(q, position);

    FREE(position_fen);
    return fault;
}

int ch_end_move(char* sessionId, char *sfen, char *fen, char* move, int idx)
{
    if (idx == -1) return 0;
    if (fen == NULL || move == NULL) return 0;
    if (fen[0] == '\0' || move[0] == '\0') return 0;
    int is_draw_in_c = ch_is_three_fold_repetition(sessionId, fen);
    return is_draw_in_c || ch_is_end(sfen, fen, idx);
}

int ch_mate_move(char* sfen, char *ko, int idx)
{
    return ch_is_checkmate_move(sfen, ko, idx);
}

float *ch_move(char* sfen, float *board, int indext) {
    char* pfen = ch_board_to_fen(board);
    char* mfen = ch_do_legal(sfen, pfen, indext);
    if (mfen == NULL) {
        FREE(pfen);
        fprintf(stderr, "try (%i) on %s\n", indext, pfen);
        return NULL;
    }
    float* mboard = ch_fen_to_board(mfen, 1);
    FREE(mfen);
    FREE(pfen);
    return mboard;
}

char *ch_move_fen(char* sfen, char* fen, int indext) {
    char* mfen = ch_do_legal(sfen, fen, indext);
    if (mfen == NULL) {
        fprintf(stderr, "try (%i) on %s\n", indext, fen);
    }
    return mfen;
}

#define BOARD_SIZE (8*8+8)

typedef struct {
    ch_nets* nets;
    char* sfen;
} ch_adapter_ctx;

static ch_adapter_ctx gctx;

static int api_gen_legal(const ch_board* board, ch_mv* moves, int max_moves) {
    char* valid_fen = NULL;
    char** valid_moves = NULL;
    int cnt = 0;
    int ok = ch_get_all_valid_moves(gctx.sfen, (char*)board->fen, &valid_fen, &valid_moves, &cnt);
    if (!ok || cnt <= 0) {
        if (valid_moves) { for (int i = 0; i < cnt; ++i) FREE(valid_moves[i]); FREE(valid_moves); }
        if (valid_fen) FREE(valid_fen);
        return 0;
    }
    int n = cnt < max_moves ? cnt : max_moves;
    for (int i = 0; i < n; ++i) moves[i] = (ch_mv)i;
    for (int i = 0; i < cnt; ++i) FREE(valid_moves[i]);
    FREE(valid_moves);
    FREE(valid_fen);
    return n;
}

static int api_apply_move(const ch_board* board, ch_mv mv, ch_board* out) {
    char* mfen = ch_move_fen(gctx.sfen, (char*)board->fen, (int)mv);
    if (!mfen) return -1;
    strncpy(out->fen, mfen, sizeof(out->fen)-1);
    out->fen[sizeof(out->fen)-1] = '\0';
    FREE(mfen);
    return 0;
}

static int api_terminal_result(const ch_board* board, int ch_player) {
    int player = strstr((char*)board->fen, " w ") ? 0 : 1;
    if (ch_is_checkmate((char*)board->fen)) return ch_player == player ? -1 : +1;
    return 0;
}

static ch_api make_api() {
    ch_api api;
    api.gen_legal = (int(*)(const ch_board*, ch_mv*, int))api_gen_legal;
    api.apply_move = (int(*)(const ch_board*, ch_mv, ch_board*))api_apply_move;
    api.terminal_result = (int(*)(const ch_board*, int))api_terminal_result;
    api.nn_eval = (int(*)(ch_nets*, const ch_board*, const ch_mv*, int, float*, float*))ch_eval_dual;
    return api;
}

void ch_extract_move_from_fen_diff(const char *fen_before, const char *fen_after, char *uci_out) {
    float *b1 = ch_fen_to_board((char*)fen_before, 1);
    float *b2 = ch_fen_to_board((char*)fen_after, 1);

    int from = -1, to = -1;
    for (int i = 0; i < 64; ++i) {
        if (b1[i] != 0 && b2[i] == 0) from = i;
        if (b1[i] == 0 && b2[i] != 0) to = i;
    }

    if (from >= 0 && to >= 0) {
        int ff = from % 8, fr = from / 8;
        int tf = to % 8, tr = to / 8;
        snprintf(uci_out, 8, "%c%d%c%d", 'a' + ff, fr + 1, 'a' + tf, tr + 1);
    } else {
        strcpy(uci_out, "0000");
    }

    FREE(b1);
    FREE(b2);
}

void ch_index_to_move(int idx, char *uci_out) {
    if (!uci_out) return;
    uci_out[0] = '\0';

    if (idx < 0 || idx >= 4672) {
        strcpy(uci_out, "0000");
        return;
    }

    int from_sq = idx / 73;
    int type = idx % 73;

    int from_rank = from_sq / 8;
    int from_file = from_sq % 8;

    int to_file = from_file;
    int to_rank = from_rank;
    char promo = 0;

    // Sliding moves (8 dirs × 7 dist = 56)
    if (type < 56) {
        int dir = type / 7;
        int dist = (type % 7) + 1;

        switch (dir) {
            case 0: to_file = from_file;         to_rank = from_rank + dist; break; // up
            case 1: to_file = from_file + dist;  to_rank = from_rank + dist; break; // up-right
            case 2: to_file = from_file + dist;  to_rank = from_rank;        break; // right
            case 3: to_file = from_file + dist;  to_rank = from_rank - dist; break; // down-right
            case 4: to_file = from_file;         to_rank = from_rank - dist; break; // down
            case 5: to_file = from_file - dist;  to_rank = from_rank - dist; break; // down-left
            case 6: to_file = from_file - dist;  to_rank = from_rank;        break; // left
            case 7: to_file = from_file - dist;  to_rank = from_rank + dist; break; // up-left
        }
    }
    // Knight moves (8)
    else if (type < 64) {
        int k = type - 56;
        static const int knight_df[8] = { +1, +2, +2, +1, -1, -2, -2, -1 };
        static const int knight_dr[8] = { +2, +1, -1, -2, -2, -1, +1, +2 };
        to_file = from_file + knight_df[k];
        to_rank = from_rank + knight_dr[k];
    }
    // Underpromotions (9)
    else {
        int promo_type = (type - 64) / 3; // 0=n,1=b,2=r
        int dir_type   = (type - 64) % 3; // 0=forward,1=left,2=right

        static const char promos[3] = { 'n', 'b', 'r' };
        promo = promos[promo_type];

        int pawn_dir = (from_rank <= 3) ? +1 : -1; // assume white if lower ranks
        to_rank = from_rank + pawn_dir;
        to_file = from_file;
        if (dir_type == 1) to_file = from_file - 1;
        if (dir_type == 2) to_file = from_file + 1;
    }

    if (to_file < 0 || to_file > 7 || to_rank < 0 || to_rank > 7) {
        strcpy(uci_out, "0000");
        return;
    }

    // Build UCI string
    uci_out[0] = 'a' + from_file;
    uci_out[1] = '1' + from_rank;
    uci_out[2] = 'a' + to_file;
    uci_out[3] = '1' + to_rank;

    if (promo) {
        uci_out[4] = promo;
        uci_out[5] = '\0';
    } else {
        uci_out[4] = '\0';
    }
}

int ch_mcts_search(const char *sfen, float tau, int sims, ch_mcts_az *out) {
    if (!sfen || !out) return -1;

    ch_mcts_config cfg = {0};
    cfg.cpuct = 1.25f;
    cfg.dirichlet_alpha = 0.03f;
    cfg.root_noise_frac = 0.10f;
    cfg.max_children = 32;
    cfg.use_virtual_loss = 0;
    cfg.virtual_loss = 0.0f;
    cfg.player = strstr(sfen, " w ") ? 0 : 1;

    static unsigned char arena[(8 * 8 + 8) * 1024 * 1024];
    ch_api api = make_api();
    ch_mcts m;
    ch_nets* nets = gctx.nets;
    ch_mcts_init(&m, api, cfg, nets, arena, sizeof(arena));

    ch_board rootb;
    strncpy(rootb.fen, sfen, sizeof(rootb.fen) - 1);
    rootb.fen[sizeof(rootb.fen) - 1] = '\0';
    int to_move = strstr(sfen, " w ") ? +1 : -1;
    ch_mcts_node *root = ch_mcts_create_root(&m, &rootb, to_move);

    ch_mcts_play_and_log(&m, root, sims, NULL);

    out->move_count = root->child_count;
    out->best_index = -1;

    float sum_q = 0.0f, sum_n = 0.0f;
    for (int i = 0; i < root->child_count; ++i) {
        sum_q += root->value_sum[i];
        sum_n += (float)root->visit_count[i];
    }
    out->q_root = (sum_n > 0.0f) ? (sum_q / sum_n) : 0.0f;

    float max_visits = -1.0f;
    for (int i = 0; i < root->child_count && i < 256; ++i) {
        out->visits[i] = (float)root->visit_count[i];
        ch_index_to_move(root->moves[i], out->moves[i]);
        if (out->visits[i] > max_visits) {
            max_visits = out->visits[i];
            out->best_index = i;
        }
    }

    return 0;
}

// convert UCI like "e2e4", "a7a8q", "g1f3" into fixed [0..4671] policy index.
int ch_move_to_index(const char *uci) {
    if (!uci || strlen(uci) < 4) return -1;

    int from_file = uci[0] - 'a';
    int from_rank = uci[1] - '1';
    int to_file   = uci[2] - 'a';
    int to_rank   = uci[3] - '1';
    if (from_file < 0 || from_file > 7 || to_file < 0 || to_file > 7 ||
        from_rank < 0 || from_rank > 7 || to_rank < 0 || to_rank > 7)
        return -1;

    int from_square = from_rank * 8 + from_file;
    int df = to_file - from_file;
    int dr = to_rank - from_rank;

    // Knight moves
    if ((abs(df) == 1 && abs(dr) == 2) || (abs(df) == 2 && abs(dr) == 1)) {
        int knight_dir = 0;
        if (df == 1 && dr == 2)  knight_dir = 0;
        if (df == 2 && dr == 1)  knight_dir = 1;
        if (df == 2 && dr == -1) knight_dir = 2;
        if (df == 1 && dr == -2) knight_dir = 3;
        if (df == -1 && dr == -2)knight_dir = 4;
        if (df == -2 && dr == -1)knight_dir = 5;
        if (df == -2 && dr == 1) knight_dir = 6;
        if (df == -1 && dr == 2) knight_dir = 7;
        return from_square * 73 + 56 + knight_dir; // 56 sliding before knights
    }

    // Sliding moves (rook/bishop/queen)
    int dir = -1;
    int distance = 0;
    if (df == 0 && dr > 0)  dir = 0; // up
    if (df > 0 && dr > 0 && abs(df) == abs(dr)) dir = 1; // up-right
    if (df > 0 && dr == 0) dir = 2; // right
    if (df > 0 && dr < 0 && abs(df) == abs(dr)) dir = 3; // down-right
    if (df == 0 && dr < 0) dir = 4; // down
    if (df < 0 && dr < 0 && abs(df) == abs(dr)) dir = 5; // down-left
    if (df < 0 && dr == 0) dir = 6; // left
    if (df < 0 && dr > 0 && abs(df) == abs(dr)) dir = 7; // up-left

    if (dir >= 0) {
        distance = (int)fmaxf(fabsf((float)df), fabsf((float)dr));
        if (distance >= 1 && distance <= 7)
            return from_square * 73 + dir * 7 + (distance - 1);
    }

    // Underpromotions (pawn reaching last rank)
    if (uci[4] != '\0') {
        int promo = 0;
        switch (uci[4]) {
            case 'n': promo = 0; break;
            case 'b': promo = 1; break;
            case 'r': promo = 2; break;
            default:  promo = 0; break;
        }
        // Direction for pawn promotions
        int pd = to_file - from_file;
        if (pd == 0) return from_square * 73 + 64 + promo * 3 + 0; // forward
        if (pd == -1) return from_square * 73 + 64 + promo * 3 + 1; // capture left
        if (pd == 1)  return from_square * 73 + 64 + promo * 3 + 2; // capture right
    }

    // Fallback: illegal or unmapped
    return -1;
}

int ch_is_capture(const char *move) { return strchr(move, 'x') != NULL; }
int ch_is_unprotected(const char *fen, const char *move) { (void)fen; (void)move; return 1; }

void ch_self_study_train_self_step(char *sessionId, char *sfen, char *valid_fen, char *valid_move, ch_nets *nets, int level, int idx, float pow, float val) {
    if (!valid_fen || !valid_move || !valid_fen[0] || !valid_move[0]) return;
    if (ch_is_checkmate(valid_fen)) return;

    int player = strstr(valid_fen, " w ") ? 0 : 1;

    float *prev = ch_fen_to_board(valid_fen, 1);
    float *next = ch_move(sfen, prev, idx);
    if (!next) { FREE(prev); return; }

    float powW = 0, powB = 0;
    float power = ch_eval_the_board(sfen, next, &powW, &powB);
    if (!isfinite(power)) power = 0;
    if (!isfinite(pow)) pow = 0;
    float value = (player == 0 ? powW - powB : powB - powW);
    if (!isfinite(value)) value = 0;

    float *x = (float *)CALLOC(2 * BOARD_SIZE, sizeof(float));
    memcpy(x, prev, BOARD_SIZE * sizeof(float));
    memcpy(x + BOARD_SIZE, next, BOARD_SIZE * sizeof(float));

    float *y = (float *)CALLOC(1, sizeof(float));
    y[0] = 0.5f * (power + pow);
    if (!isfinite(y[0])) y[0] = 0.f;

    float *pi_full = (float*)CALLOC(CH_ACTION_SPACE, sizeof(float));
    int k_move = ch_move_to_index(valid_move);
    if (k_move >= 0 && k_move < CH_ACTION_SPACE) pi_full[k_move] = 1.0f;

    float capture_bonus = 0.f;
    if (ch_is_capture(valid_move)) {
        capture_bonus = ch_is_unprotected(valid_fen, valid_move) ? 0.15f : 0.08f;
    }

    float loss = ch_train_dual(nets, x, pi_full, y[0], capture_bonus);

    if (loss != loss) {
        fprintf(stderr,
            "train step %ld(%s): sub-step: (%i) step-level: (%i) train: (%i) rate: (%.8g) loss: (%.8g)\n",
            nets->netv->nsteps, player == 0 ? "w" : "b",
            ch_queue_count(ch_dict_get(moves_history, sessionId)) + 1,
            level, idx, nets->netv->learning_rate, loss);
        fprintf(stderr, "\nNaN LOSS detected! No possible to continue!\n");
        exit(4);
    }

    FREE(pi_full);
    FREE(y);
    FREE(x);
    FREE(next);
    FREE(prev);
}

float *ch_copy_board(float *board)
{
    float *next = (float*) CALLOC(BOARD_SIZE, sizeof(float));
    memcpy(next, board, (BOARD_SIZE)*sizeof(float));
    return next;
}

void ch_swap(float* b, int i, int j) {
    float swap = b[i];
    b[i] = b[j];
    b[j] = swap;
}

void ch_flip_board(float *board)
{
    int i;
    for(i = 0; i < 8*8; ++i) {
        board[i] = -1 * board[i];
    }
    board[8*8] = board[8*8] == 0 ? 1 : 0;
    ch_swap(board, 8*8+1, 8*8+3);
    ch_swap(board, 8*8+2, 8*8+4);
    board[8*8+5] = board[8*8+5] == 0 ? 0.f : (float)(64 - (int)board[8*8+5]);
    ch_swap(board, 8*8+6, 8*8+7);
}

void ch_put_into_game_queue(char *sessionId, char *valid_fen, char* valid_move, int indext, ch_nets* nets, char* sfen) {
    int player = strstr(valid_fen, " w ") ? 0 : 1;
    ch_constant_memory_queue *q = ch_dict_get(moves_history, sessionId);
    ch_learn_state* to_learn = (ch_learn_state *) CALLOC(1, sizeof(ch_learn_state));
    float* fen_board = ch_fen_to_board(valid_fen, 1);
    memcpy(to_learn->board, fen_board, (1)*(BOARD_SIZE)*sizeof(float));
    FREE(fen_board);
    to_learn->index = indext;
    to_learn->player = player;
    ch_put_back(sessionId, to_learn);
}

int ch_pick_move_mcts(char* sessionId, char* sfen, char* valid_fen, char** valid_moves, ch_moves ch_m, ch_nets* nets, int n, int level, int *solver, float *pow, float *val)
{
    gctx.nets = nets;
    gctx.sfen = sfen;
    ch_mcts_config cfg;
    cfg.cpuct = 0.72f;
    cfg.dirichlet_alpha = 0.03f;
    cfg.root_noise_frac = 0.05f;
    cfg.max_children = 32;
    cfg.use_virtual_loss = 0;
    cfg.virtual_loss = 0.1f;
    cfg.player = strstr((char*)valid_fen, " w ") ? 0 : 1;
    static unsigned char arena[(8*8+8) * 1024 * 1024];
    ch_mcts m;
    ch_api api = make_api();
    ch_mcts_init(&m, api, cfg, nets, arena, sizeof(arena));
    ch_board rootb;
    strncpy(rootb.fen, valid_fen, sizeof(rootb.fen)-1);
    rootb.fen[sizeof(rootb.fen)-1] = '\0';
    int to_move = strstr(valid_fen, " w ") ? +1 : -1;
    ch_mcts_node* root = ch_mcts_create_root(&m, &rootb, to_move);
    int sims = (level < 0 ? 0 : level) + 1;
    ch_print_board(valid_fen);
#ifdef CH_ENGINE
    ch_mcts_play_and_log(&m, root, 1, valid_moves);
#else
    ch_mcts_play_and_log(&m, root, level + 1, valid_moves);
#endif
    memset(&g_last_res, 0, sizeof(g_last_res));
    g_last_res.move_count = root->child_count;
    g_last_res.best_index = -1;
    {
        float sum_q = 0.0f, sum_n = 0.0f, max_v = -1.0f;
        for (int idx = 0; idx < root->child_count && idx < 256; ++idx) {
            float n_n = (float)root->visit_count[idx];
            float q_sum = root->value_sum[idx];
            g_last_res.visits[idx] = n_n;
            ch_index_to_move(root->moves[idx], g_last_res.moves[idx]);
            if (n_n > max_v) { max_v = n_n; g_last_res.best_index = idx; }
            sum_q += q_sum;
            sum_n += n_n;
        }
        g_last_res.q_root = (sum_n > 0.0f) ? (sum_q / sum_n) : 0.0f;
    }
    ch_mv mv = ch_mcts_pick_move(&m, root, 0.0f);
    int idx = (int)mv;
    if (idx < 0 || idx >= n) idx = 0;
    *solver = 2;
    float* prev = ch_fen_to_board(valid_fen, 1);
    float* next = ch_move(sfen, prev, idx);
    float powW=0.f, powB=0.f;
    ch_eval_the_board(sfen, next, &powW, &powB);
    if (strstr(valid_fen, " w ")) { *pow = powW; *val = powW - powB; } else { *pow = powB; *val = powB - powW; }
    FREE(prev);
    return idx;
}

static char *ch_lin_in_dir;
static char *ch_lin_out_dir;
static ch_nets *ch_lin_nets;

#ifndef __linux__

int ch_process_file(char *file_name) {

    ch_nets* nets  = ch_lin_nets;
    if (++nets->netp->nsteps % 1000 == 0) save_weights(nets->netp, nets->netp_wname);
    if (++nets->netv->nsteps % 1000 == 0) save_weights(nets->netv, nets->netv_wname);

    char* in_dir  = ch_lin_in_dir;
    char* out_dir = ch_lin_out_dir;

    fprintf(stderr, "fn: %s\n", file_name);

    char fname[1024];
    char ffiname[1024];
    char ffoname[1024];

    strcpy(fname, file_name);

    strcpy(ffiname, ch_lin_in_dir);
    strcat(ffiname, "/");
    strcat(ffiname, fname);
    fname[strlen(ch_lin_in_dir) + strlen("/") + strlen(fname) + 1] = '\0';

    strcpy(ffoname, ch_lin_out_dir);
    strcat(ffoname, "/");
    strcat(ffoname, fname);
    fname[strlen(ch_lin_out_dir) + strlen("/") + strlen(fname) + 1] = '\0';

    struct stat ch_st = {0};
    off_t size = 0;
    off_t offs = 0;
    do {
        offs = size;
        stat(ffiname, &ch_st);
        size = ch_st.st_size;
        if (offs != size) usleep(250); else break;
    } while (1);

    char* pw = " w ";
    char* pb = " b ";

    char* sessionId = NULL;
    char* fen = NULL;
    char* fen_move = NULL;
    char* level = NULL;
    char* sfen = NULL;
    int mlevel = 0;

    int solver = 0;

    if (ch_fopen(ffiname, &sessionId, &fen, &fen_move, &level, &sfen)) {

        if (fen == NULL || fen[0] == '\0') {
            if (fen) FREE(fen);
            fen = CALLOC(strlen(sfen) + 1, sizeof(char));
            strcpy(fen, sfen);
        }

        fprintf(stderr, "sessionId: %s\n", sessionId);
        fprintf(stderr, "level: %s\n", level);
        fprintf(stderr, "sfen: %s\n", sfen);
        fprintf(stderr, "fen: %s\n", fen);

        ch_init_game_history(sessionId);

        if (fen_move != NULL && strcmp(fen_move, "") != 0) {
            fprintf(stderr, "pgn: %s\n", fen_move);
            char *fen_next = NULL;
            int fen_next_idx = 0;
            int fen_next_cnt = 0;
            if (ch_board_after_move(sfen, fen, fen_move, &fen_next, &fen_next_idx, &fen_next_cnt)) {
                if (fen_next == NULL) {
                    ch_fsave(ffoname, sessionId, NULL, NULL, level, sfen, solver);
                    return 1;
                }
                ch_print_board(fen_next);
                FREE(fen_next);
            }
        }

        mlevel = level != NULL ? (int)atoi(level) : 3;

        int valid_moves_count = 0;
        char **valid_moves = NULL;
        char* valid_fen = NULL;

        if (fen != NULL && fen[0] != '\0' && ch_get_all_valid_moves(sfen, fen, &valid_fen, &valid_moves, &valid_moves_count)) {

            int indext = -1;
            int player = strstr(valid_fen, " w ") ? 0 : 1;

            nets->netp->nsteps++;
            nets->netv->nsteps++;

            if (indext == -1 && fen_move != NULL) {
                for (int i = 0; i < valid_moves_count; ++i) {
                    if (strcmp(valid_moves[i], fen_move) == 0) {
                        indext = i;
                        break;
                    }
                }
            }

            if ((indext != -1) && ch_end_move(sessionId, sfen, fen, valid_moves[indext], indext)) {
                ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, nets, sfen);
                ch_fsave(ffoname, sessionId, valid_fen, valid_moves[indext], level, sfen, solver);
                ch_clean_history(sessionId, 1);
                for (int j = 0; j < valid_moves_count; ++j) FREE(valid_moves[j]);
                FREE(valid_moves);
                FREE(valid_fen);
                return 0;
            }

            if (indext == -1) {
                ch_moves ch_m = ch_load_moves(valid_fen, valid_moves, valid_moves_count);
                float pow = 0.f;
                float value = 0.f;
                indext = ch_pick_move_mcts(sessionId, sfen, valid_fen, valid_moves, ch_m, nets, valid_moves_count, mlevel, &solver, &pow, &value);
                ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, nets, sfen);
                ch_self_study_train_self_step(sessionId, sfen, valid_fen, valid_moves[indext], nets, mlevel, indext, pow, value);
                ch_fsave(ffoname, sessionId, valid_fen, valid_moves[indext], level, sfen, solver);
                for (int j = 0; j < valid_moves_count; ++j) FREE(valid_moves[j]);
                FREE(valid_moves);
                FREE(valid_fen);
                return 0;
            }

            //ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, nets, sfen);

            strcpy(fen, valid_fen);
            if (fen_move != NULL) strcpy(fen_move, valid_moves[indext]);

            FREE(valid_fen); valid_fen = NULL;
            for (int i = 0; i < valid_moves_count; ++i) {FREE(valid_moves[i]); valid_moves[i] = NULL; } valid_moves_count = 0;
            FREE(valid_moves); valid_moves = NULL;

            valid_moves_count = 0;
            valid_moves = NULL;
            valid_fen = NULL;

            if (ch_get_all_valid_moves_after(sfen, fen, fen_move, &valid_fen, &valid_moves, &valid_moves_count)) {

                indext = -1;
                player = strstr(valid_fen, " w ") ? 0 : 1;

                ch_moves ch_m = ch_load_moves(valid_fen, valid_moves, valid_moves_count);
                float pow = 0.f;
                float value = 0.f;
                indext = ch_pick_move_mcts(sessionId, sfen, valid_fen, valid_moves, ch_m, nets, valid_moves_count, mlevel, &solver, &pow, &value);
                ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, nets, sfen);
                ch_self_study_train_self_step(sessionId, sfen, valid_fen, valid_moves[indext], nets, mlevel, indext, pow, value);

                if (valid_moves[indext] != NULL && strcmp(valid_moves[indext], "") != 0) {
                    char *fen_next = NULL;
                    int fen_next_idx = 0;
                    int fen_next_cnt = 0;
                    if (ch_board_after_move(sfen, valid_fen, valid_moves[indext], &fen_next, &fen_next_idx,&fen_next_cnt)) {
                        FREE(fen_next);
                    }
                }

                ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, nets, sfen);
                ch_fsave(ffoname, sessionId, valid_fen, valid_moves[indext], level, sfen, solver);

                FREE(valid_fen);
                for (int i = 0; i < valid_moves_count; ++i) FREE(valid_moves[i]);
                FREE(valid_moves);

            } else {
                ch_fsave(ffoname, sessionId, NULL, NULL, level, sfen, solver);
            }
        }

        FREE(sfen);
        FREE(fen_move);
        FREE(fen);
        FREE(sessionId);
    }

    remove(ffiname);

    if (++nets->netp->nsteps % 1000 == 0) save_weights(nets->netp, nets->netp_wname);
    if (++nets->netv->nsteps % 1000 == 0) save_weights(nets->netv, nets->netv_wname);

    return 0;
}

void test_dchess(int argc, char **argv, char *cfgfile, char *weight_file, char *in_dir, char *out_dir) {
    srandom(time(0));

    ch_nets *nets = ch_load_networks(cfgfile, weight_file);

    ch_init_game_history("13a25e80-ece3-4a4b-9347-e6df74386d02");

    set_batch_network(nets->netp, 1);
    set_batch_network(nets->netv, 1);
    ch_lin_nets = nets;

    ch_lin_in_dir = in_dir;
    ch_lin_out_dir = out_dir;

    const char* patterns[] = {"*.json"};
    while (!init_notified_file_name(in_dir, patterns, ch_process_file));
}

#else

int ch_process_file(char *file_name) {
    ch_nets* nets  = ch_lin_nets;
    if (++nets->netp->nsteps % 1000 == 0) save_weights(nets->netp, nets->netp_wname);
    if (++nets->netv->nsteps % 1000 == 0) save_weights(nets->netv, nets->netv_wname);

    char* in_dir  = ch_lin_in_dir;
    char* out_dir = ch_lin_out_dir;

    fprintf(stderr, "fn: %s\n", file_name);

    char fname[1024];
    char ffiname[1024];
    char ffoname[1024];

    strcpy(fname, file_name);

    strcpy(ffiname, ch_lin_in_dir);
    strcat(ffiname, "/");
    strcat(ffiname, fname);
    fname[strlen(ch_lin_in_dir) + strlen("/") + strlen(fname) + 1] = '\0';

    strcpy(ffoname, ch_lin_out_dir);
    strcat(ffoname, "/");
    strcat(ffoname, fname);
    fname[strlen(ch_lin_out_dir) + strlen("/") + strlen(fname) + 1] = '\0';

    struct stat ch_st = {0};
    off_t size = 0;
    off_t offs = 0;
    do {
        offs = size;
        stat(ffiname, &ch_st);
        size = ch_st.st_size;
        if (offs != size) usleep(500); else break;
    } while (1);

    char* pw = " w ";
    char* pb = " b ";

    char* sessionId = NULL;
    char* fen = NULL;
    char* fen_move = NULL;
    char* level = NULL;
    char* sfen = NULL;
    int mlevel = 0;

    int solver = 0;

    if (ch_fopen(ffiname, &sessionId, &fen, &fen_move, &level, &sfen)) {

        if (fen == NULL || fen[0] == '\0') {
            if (fen) FREE(fen);
            fen = CALLOC(strlen(sfen) + 1, sizeof(char));
            strcpy(fen, sfen);
        }

        fprintf(stderr, "sessionId: %s\n", sessionId);
        fprintf(stderr, "level: %s\n", level);
        fprintf(stderr, "sfen: %s\n", sfen);
        fprintf(stderr, "fen: %s\n", fen);

        ch_init_game_history(sessionId);

        if (fen_move != NULL && strcmp(fen_move, "") != 0) {
            fprintf(stderr, "pgn: %s\n", fen_move);
            char *fen_next = NULL;
            int fen_next_idx = 0;
            int fen_next_cnt = 0;
            if (ch_board_after_move(sfen, fen, fen_move, &fen_next, &fen_next_idx, &fen_next_cnt)) {
                if (fen_next == NULL) {
                    ch_fsave(ffoname, sessionId, NULL, NULL, level, sfen, solver);
                    return 1;
                }
                ch_print_board(fen_next);
                FREE(fen_next);
            }
        }

        mlevel = level != NULL ? (int)atoi(level) : 3;

        int valid_moves_count = 0;
        char **valid_moves = NULL;
        char* valid_fen = NULL;

        if (fen != NULL && fen[0] != '\0' && ch_get_all_valid_moves(sfen, fen, &valid_fen, &valid_moves, &valid_moves_count)) {

            int indext = -1;
            int player = strstr(valid_fen, " w ") ? 0 : 1;

            nets->netp->nsteps++;
            nets->netv->nsteps++;

            if (indext == -1 && fen_move != NULL) {
                for (int i = 0; i < valid_moves_count; ++i) {
                    if (strcmp(valid_moves[i], fen_move) == 0) {
                        indext = i;
                        break;
                    }
                }
            }

            if ((indext != -1) && ch_end_move(sessionId, sfen, fen, valid_moves[indext], indext)) {
                ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, nets, sfen);
                ch_fsave(ffoname, sessionId, valid_fen, valid_moves[indext], level, sfen, solver);
                ch_clean_history(sessionId, 1);
                for (int j = 0; j < valid_moves_count; ++j) FREE(valid_moves[j]);
                FREE(valid_moves);
                FREE(valid_fen);
                return 0;
            }

            if (indext == -1) {
                ch_moves ch_m = ch_load_moves(valid_fen, valid_moves, valid_moves_count);
                float pow = 0.f;
                float value = 0.f;
                indext = ch_pick_move_mcts(sessionId, sfen, valid_fen, valid_moves, ch_m, nets, valid_moves_count, mlevel, &solver, &pow, &value);
                ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, nets, sfen);
                ch_self_study_train_self_step(sessionId, sfen, valid_fen, valid_moves[indext], nets, mlevel, indext, pow, value);
                ch_fsave(ffoname, sessionId, valid_fen, valid_moves[indext], level, sfen, solver);
                for (int j = 0; j < valid_moves_count; ++j) FREE(valid_moves[j]);
                FREE(valid_moves);
                FREE(valid_fen);
                return 0;
            }

            //ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, nets, sfen);

            strcpy(fen, valid_fen);
            if (fen_move != NULL) strcpy(fen_move, valid_moves[indext]);

            FREE(valid_fen); valid_fen = NULL;
            for (int i = 0; i < valid_moves_count; ++i) {FREE(valid_moves[i]); valid_moves[i] = NULL; } valid_moves_count = 0;
            FREE(valid_moves); valid_moves = NULL;

            valid_moves_count = 0;
            valid_moves = NULL;
            valid_fen = NULL;

            if (ch_get_all_valid_moves_after(sfen, fen, fen_move, &valid_fen, &valid_moves, &valid_moves_count)) {

                indext = -1;
                player = strstr(valid_fen, " w ") ? 0 : 1;

                ch_moves ch_m = ch_load_moves(valid_fen, valid_moves, valid_moves_count);
                float pow = 0.f;
                float value = 0.f;
                indext = ch_pick_move_mcts(sessionId, sfen, valid_fen, valid_moves, ch_m, nets, valid_moves_count, mlevel, &solver, &pow, &value);
                ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, nets, sfen);
                ch_self_study_train_self_step(sessionId, sfen, valid_fen, valid_moves[indext], nets, mlevel, indext, pow, value);

                if (valid_moves[indext] != NULL && strcmp(valid_moves[indext], "") != 0) {
                    char *fen_next = NULL;
                    int fen_next_idx = 0;
                    int fen_next_cnt = 0;
                    if (ch_board_after_move(sfen, valid_fen, valid_moves[indext], &fen_next, &fen_next_idx, &fen_next_cnt)) {
                        FREE(fen_next);
                    }
                }

                ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, nets, sfen);
                ch_fsave(ffoname, sessionId, valid_fen, valid_moves[indext], level, sfen, solver);

                FREE(valid_fen);
                for (int i = 0; i < valid_moves_count; ++i) FREE(valid_moves[i]); valid_moves_count = 0;
                FREE(valid_moves);

            } else {
                ch_fsave(ffoname, sessionId, NULL, NULL, level, sfen, solver);
            }
        }

        FREE(sfen);
        FREE(fen_move);
        FREE(fen);
        FREE(sessionId);
    }

    remove(ffiname);

    return 0;
}

void test_dchess(int argc, char **argv, char *cfgfile, char *weight_file, char *in_dir, char *out_dir) {
    srandom(time(0));

    ch_nets *nets = ch_load_networks(cfgfile, weight_file);

    ch_init_game_history("13a25e80-ece3-4a4b-9347-e6df74386d02");

    set_batch_network(nets->netp, 1);
    set_batch_network(nets->netv, 1);
    ch_lin_nets = nets;

    ch_lin_in_dir = in_dir;
    ch_lin_out_dir = out_dir;

    const char* patterns[] = {"*.json"};
    while (!init_notified_file_name(in_dir, patterns, ch_process_file));
}

#endif

typedef struct ch_board_state {
    char fen[128];
    char move[16];
    int final;
    int indext;
} ch_board_state;

ch_board_state ch_self_learn_step(char* sessionId, char* sfen, int level, ch_nets* nets, char *fen, char *fen_move, int learn) {

    int player = strstr(fen, " w ") ? 0 : 1;

    ch_board_state return_value = {0};
    return_value.final = 0;

    int valid_moves_count = 0;
    char **valid_moves = NULL;
    char* valid_fen = NULL;

    if (ch_get_all_valid_moves(sfen, fen, &valid_fen, &valid_moves, &valid_moves_count)) {

        int indext = -1;

        if (indext == -1 && fen_move != NULL && fen_move[0] != '\0') {
            for (int i = 0; i < valid_moves_count; ++i) {
                if (strcmp(valid_moves[i], fen_move) == 0) {
                    indext = i;
                    break;
                }
            }
        }

        if ((indext != -1) && ch_end_move(sessionId, sfen, fen, valid_moves[indext], indext)) {
            ch_put_into_game_queue(sessionId, fen, valid_moves[indext], indext, nets, sfen);
            strcpy(return_value.fen, valid_fen);
            strcpy(return_value.move, valid_moves[indext]);
            return_value.indext = indext;
            for (int j = 0; j < valid_moves_count; ++j) FREE(valid_moves[j]);
            FREE(valid_moves);
            FREE(valid_fen);
            return_value.final = 1;
            return return_value;
        }

        if (indext == -1) {
            ch_moves ch_m = ch_load_moves(valid_fen, valid_moves, valid_moves_count);
            int solver = 0;
            float pow = 0.f;
            float value = 0.f;
            indext = ch_pick_move_mcts(sessionId, sfen, valid_fen, valid_moves, ch_m, nets, valid_moves_count, level, &solver, &pow, &value);
            ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, nets, sfen);
            if (learn == 1) ch_self_study_train_self_step(sessionId, sfen, valid_fen, valid_moves[indext], nets, level, indext, pow, value);
            strcpy(return_value.fen, valid_fen);
            strcpy(return_value.move, valid_moves[indext]);
            return_value.indext = indext;
            for (int j = 0; j < valid_moves_count; ++j) FREE(valid_moves[j]);
            FREE(valid_moves);
            FREE(valid_fen);
            return return_value;
        }

        strcpy(fen, valid_fen);
        strcpy(fen_move, valid_moves[indext]);

        for (int j = 0; j < valid_moves_count; ++j) FREE(valid_moves[j]);
        FREE(valid_moves);
        FREE(valid_fen);

        valid_moves_count = 0;
        valid_moves = NULL;
        valid_fen = NULL;

        if (ch_get_all_valid_moves_after(sfen, fen, fen_move, &valid_fen, &valid_moves, &valid_moves_count)) {

            indext = -1;
            player = strstr(valid_fen, " w ") ? 0 : 1;

            ch_moves ch_m = ch_load_moves(valid_fen, valid_moves, valid_moves_count);
            int solver = 0;
            float pow = 0.f;
            float value = 0.f;
            indext = ch_pick_move_mcts(sessionId, sfen, valid_fen, valid_moves, ch_m, nets, valid_moves_count, level, &solver, &pow, &value);
            ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, nets, sfen);
            if (learn == 1) ch_self_study_train_self_step(sessionId, sfen, valid_fen, valid_moves[indext], nets, level, indext, pow, value);
            strcpy(return_value.fen, valid_fen);
            strcpy(return_value.move, valid_moves[indext]);
            return_value.indext = indext;
            FREE(valid_fen); valid_fen = NULL;
            for (int i = 0; i < valid_moves_count; ++i) {FREE(valid_moves[i]); valid_moves[i] = NULL; }
            FREE(valid_moves); valid_moves = NULL;
            return return_value;
        }

        strcpy(return_value.fen, valid_fen);
        strcpy(return_value.move, valid_moves[indext]);
        return_value.indext = indext;

        FREE(valid_fen); valid_fen = NULL;
        for (int i = 0; i < valid_moves_count; ++i) {FREE(valid_moves[i]); valid_moves[i] = NULL; }
        FREE(valid_moves); valid_moves = NULL;
    }

    return return_value;
}

void test_tchess(int argc, char **argv, char *cfgfile, char *weight_file) {
    trivial_player = 0;
    srandom(time(0));

    ch_nets* nets = ch_load_networks(cfgfile, weight_file);

    char* sessionId = "13a25e80-ece3-4a4b-9347-e6df74386d02";

    ch_init_game_history(sessionId);

    char *ch_weight_file = weight_file;

    char valid_fen[128];
    char valid_fen_move[8];
    char sfen[128];

    ch_board_state move_state = {0};

    int mcount = 0;
    int mlevel = 0;

    char* startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    if (0) {
        startpos = "rnbqkbnr/pp2pppp/8/2ppP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3";
        float* fboard = ch_fen_to_board(startpos, 1);
        char* cfen = ch_board_to_fen(fboard);
        if (strcmp(cfen, startpos) != 0) {
            fprintf(stderr, "%s\n", startpos);
            ch_print_board(startpos);
            float *board = ch_fen_to_board(startpos, 1);
            char *fen = ch_board_to_fen(board);
            fprintf(stderr, "%s\n", fen);
            FREE(fen);
            FREE(board);
            exit(1);
        }
        FREE(cfen);
        FREE(fboard);
    }

    if (0) {
        startpos = "rnbqkbnr/pp2pppp/8/2ppP3/8/8/PPPP1PPP/RNBQKBNR w HAha d6 0 3";
        float* fboard = ch_fen_to_board(startpos, 1);
        char* cfen = ch_board_to_fen(fboard);
        if (strcmp(cfen, startpos) != 0) {
            fprintf(stderr, "%s\n", startpos);
            ch_print_board(startpos);
            float *board = ch_fen_to_board(startpos, 1);
            char *fen = ch_board_to_fen(board);
            fprintf(stderr, "%s\n", fen);
            FREE(fen);
            FREE(board);
            exit(1);
        }
        FREE(cfen);
        FREE(fboard);
    }

    if (0) {
        char* test_fen = "1kr5/3n4/q3p2p/p2n2p1/PppB1P2/5BP1/1P2Q2P/3R2K1 w - - 0 1";
        move_state = ch_self_learn_step(sessionId, sfen, 24, nets, test_fen, "", 0);
        assert(strcmp("f4f5", move_state.move) == 0);
    }

    int player = 0;
    trivial_player = 0;
    one_more_time:
    player = 0;
    ch_clean_history(sessionId, 1);
    strcpy(valid_fen, startpos);
    strcpy(valid_fen_move, "");
    if (++ch_test_tchess_count % 27 != 0) {
        char* fen960 = ch_get_fen_960();
        strcpy(valid_fen, fen960);
        FREE(fen960);
    }
    mlevel = nets->netv->nsteps < nets->netv->burn_in ? 0 : 4;
    strcpy(sfen, valid_fen);
    while (nets->netv->nsteps < nets->netv->max_batches) {
        move_state = ch_self_learn_step(sessionId, sfen, mlevel, nets, valid_fen, valid_fen_move, 1);
        strcpy(valid_fen, move_state.fen);
        strcpy(valid_fen_move, move_state.move);
        if (ch_end_move(sessionId, sfen, valid_fen, valid_fen_move, move_state.indext)) {
            ch_put_into_game_queue(sessionId, valid_fen, valid_fen_move, move_state.indext, nets, sfen);
            save_weights(nets->netv, nets->netv_wname);
            save_weights(nets->netp, nets->netp_wname);
            move_state.final = 0;
            player = player == 0 ? 1 : 0;
            if (trivial_player != -1) trivial_player = trivial_player == 0 ? 1 : 0;
            goto one_more_time;
        }
        float* board = ch_fen_to_board(valid_fen, 1);
        float* next = ch_move(sfen, board, move_state.indext);
        if (next) {
            char* next_fen = ch_board_to_fen(next);
            strcpy(move_state.fen, next_fen);
            FREE(next_fen);
        }
        FREE(next);
        FREE(board);
        if (++nets->netv->nsteps % 10000 == 0) save_weights(nets->netv, nets->netv_wname);
        if (++nets->netp->nsteps % 10000 == 0) save_weights(nets->netp, nets->netp_wname);
        if (move_state.final == 1) {
            move_state.final = 0;
            player = player == 0 ? 1 : 0;
            if (trivial_player != -1) trivial_player = trivial_player == 0 ? 1 : 0;
            goto one_more_time;
        }
    }
    ch_free_networks(nets);
}

void test_echess(int argc, char** argv, char *cfgfile, char *weight_file) {
    char valid_fen[128];
    char valid_fen_move[8];
    char sfen[128];
    char valid_fen_next[128];
    char valid_fen_last[128];

    ch_board_state move_state = {0};

    int print = 0;

    char* startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    int level = 0;

    ch_nets* nets = ch_load_networks(cfgfile, weight_file);

    char* sessionId = "13a25e80-ece3-4a4b-9347-e6df74386d02";

    ch_init_game_history(sessionId);

    FILE* log = NULL;
    if (print) log = fopen("log.txt", "a+");

    fprintf(stdout, "iChess.io by Piotr Sowa v7.27\n");

    strcpy(valid_fen, startpos);
    strcpy(valid_fen_move, "");
    //ch_get_fen_960(valid_fen);
    strcpy(sfen, valid_fen);

    int player = 0;
    ch_clean_history(sessionId, 1);

    char *buff = NULL;
    size_t len = 0;
    size_t nread = 0;
    while ((nread = getline(&buff, &len, stdin)) > 0) {
        if (buff == NULL) continue;
        buff[strcspn(buff, "\r\n")] = 0;

        if (print) {
            fprintf(log, "%s\n", buff);
            fflush(log);
        }
        if (strncmp(buff, "ucinewgame", 10) == 0) {
            strcpy(move_state.fen, valid_fen);
            strcpy(move_state.move, valid_fen_move);
            strcpy(valid_fen, sfen[0] != '\0' ? sfen : startpos);
            strcpy(valid_fen_move, "");
            strcpy(sfen, "");
            player = 0;
            ch_clean_history(sessionId, 1);
			save_weights(nets->netv, nets->netv_wname);
            save_weights(nets->netp, nets->netp_wname);
            fprintf(stdout, "%s\n", "uciok");
            fflush(stdout);

        }
        else if (strncmp(buff, "uci", 3) == 0) {
            strcpy(valid_fen, startpos);
            strcpy(valid_fen_move, "");
            fprintf(stdout, "%s\n", "id name iChess.io 7.27");
            fprintf(stdout, "%s\n", "id author Piotr Sowa");
            fprintf(stdout, "%s\n", "option name UCI_Chess960 type check default false");
            fprintf(stdout, "%s\n", "option name BackendOptions type string default");
            fprintf(stdout, "%s\n", "option name Ponder type check default false");
            fprintf(stdout, "%s\n", "option name MultiPV type spin default 1 min 1 max 500");
            fprintf(stdout, "%s\n", "uciok");
            fflush(stdout);

        }
        else if (strncmp(buff, "isready", 7) == 0) {
            fprintf(stdout,"readyok\n");
            fflush(stdout);

        }
        else if (strncmp(buff, "stop", 4) == 0) {
            // ;-)
        }
        else if (strncmp(buff, "quit", 4) == 0) {
            break;
        }
        else if (strncmp(buff, "position ", 9) == 0){
            char** moves = NULL;
            char* move = NULL;
            char* sfenn = NULL;
            char* mfenn = NULL;

            int count = 0;
            char* fen = ch_analyze_pos(sfen, buff, &sfenn, &mfenn, &moves, &move, &count);

            strcpy(sfen, sfenn != NULL ? sfenn : "");
            strcpy(valid_fen, fen != NULL ? fen : "");
            strcpy(valid_fen_move, move != NULL ? move : "");
            strcpy(valid_fen_next, sfenn != NULL ? sfenn : "");
            strcpy(valid_fen_last, mfenn != NULL ? mfenn : "");

            char *mfen_next = NULL;
            int mfen_next_idx = 0;
            int mfen_next_cnt = 0;

            int qcount = ch_queue_count(ch_dict_get(moves_history,sessionId));
            int exists = qcount > 1;

            player = qcount % 2 == 0 ? 1 : 0;

            int addon = (exists ? count : qcount);

            if (exists) {

                if (ch_board_after_move(sfen, valid_fen_last, moves[count - 1], &mfen_next, &mfen_next_idx, &mfen_next_cnt)) {

                    strcpy(move_state.fen, valid_fen_last);
                    strcpy(move_state.move, moves[count - 1]);
                    move_state.indext = mfen_next_idx;

                    //ch_put_into_game_queue(sessionId, valid_fen_last, moves[count - 1], mfen_next_idx, nets, sfen);

                    float *board0 = ch_fen_to_board(valid_fen_last, 1);
                    float *board_next0 = ch_fen_to_board(mfen_next, 1);
                    float pow0 = ch_eval_the_move(sfen, valid_fen_last, mfen_next);
                    float powW0 = 0;
                    float powB0 = 0;
                    ch_eval_the_board(sfen, board0, &powW0, &powB0);
                    float value0 = player == 0 ? powW0 : powB0;

                    fprintf(stderr,
                            "pick: step %ld(%s): count: (%i) checked: (%i) power: (%.7f) index: (%i)\n",
                            nets->netv->nsteps, player == 0 ? "w" : "b", mfen_next_cnt, 1, pow0, mfen_next_idx);

                    ch_self_study_train_self_step(sessionId, sfen, valid_fen_last, moves[count - 1], nets, level, mfen_next_idx, pow0, value0);

                    FREE(board0);
                    FREE(board_next0);

                }
				FREE(mfen_next);
            }
            else if (count > 0) {

                if (ch_board_after_move(sfen, valid_fen_next, moves[addon - 1], &mfen_next, &mfen_next_idx, &mfen_next_cnt)) {

                    strcpy(move_state.fen, valid_fen_next);
                    strcpy(move_state.move, moves[addon - 1]);
                    move_state.indext = mfen_next_idx;

                    ch_put_into_game_queue(sessionId, valid_fen_next, moves[addon - 1], mfen_next_idx, nets, sfen);

                    if (++nets->netv->nsteps % 1000 == 0) save_weights(nets->netv, nets->netv_wname);
                    if (++nets->netp->nsteps % 1000 == 0) save_weights(nets->netp, nets->netp_wname);

                    float *board1 = ch_fen_to_board(valid_fen_next, 1);
                    float *board_next1 = ch_fen_to_board(mfen_next, 1);
                    float pow1 = ch_eval_the_move(sfen, valid_fen_next, mfen_next);
                    float powW1 = 0;
                    float powB1 = 0;
                    ch_eval_the_board(sfen, board1, &powW1, &powB1);
                    float value1 = player == 0 ? powW1 : powB1;

                    fprintf(stderr,
                            "pick: step %ld(%s): count: (%i) checked: (%i) power: (%.7f) index: (%i)\n",
                            nets->netv->nsteps, player == 0 ? "w" : "b", mfen_next_cnt, 1, pow1, mfen_next_idx);

                    ch_self_study_train_self_step(sessionId, sfen, valid_fen_next, moves[addon - 1], nets, level, mfen_next_idx, pow1, value1);

                    strcpy(valid_fen_next, mfen_next);
                    strcpy(valid_fen_last, mfen_next);

                    strcpy(valid_fen, valid_fen_next);
                    strcpy(valid_fen_move, moves[addon - 1]);

                    addon = (exists ? count : ch_queue_count(ch_dict_get(moves_history, sessionId)));

                    for (int i = addon - 1; i < count; ++i) {

                        if (ch_board_after_move(sfen, valid_fen_next, moves[addon - 1], &mfen_next, &mfen_next_idx, &mfen_next_cnt)) {

                            strcpy(move_state.fen, valid_fen_next);
                            strcpy(move_state.move, moves[addon - 1]);
                            move_state.indext = mfen_next_idx;

                            ch_put_into_game_queue(sessionId, valid_fen_next, moves[addon - 1], mfen_next_idx, nets , sfen);

                            if (++nets->netv->nsteps % 1000 == 0) save_weights(nets->netv, nets->netv_wname);
                            if (++nets->netp->nsteps % 1000 == 0) save_weights(nets->netp, nets->netp_wname);

                            float *board2 = ch_fen_to_board(valid_fen_next, 1);
                            float *board_next2 = ch_fen_to_board(mfen_next, 1);
                            float pow2 = ch_eval_the_move(sfen, valid_fen_next, mfen_next);
                            float powW2 = 0;
                            float powB2 = 0;
                            ch_eval_the_board(sfen, board2, &powW2, &powB2);
                            float value2 = player == 0 ? powW2 : powB2;

                            fprintf(stderr,
                                    "pick: step %ld(%s): count: (%i) checked: (%i) power: (%.7f) index: (%i)\n",
                                    nets->netv->nsteps, player == 0 ? "w" : "b", mfen_next_cnt, 1, pow2, mfen_next_idx);

                            ch_self_study_train_self_step(sessionId, sfen, valid_fen_next, moves[addon - 1], nets, level,mfen_next_idx, pow2, value2);

                            strcpy(valid_fen_next, mfen_next);
                            strcpy(valid_fen_last, mfen_next);

                            strcpy(valid_fen, valid_fen_next);
                            strcpy(valid_fen_move, moves[addon - 1]);

                            addon = (exists ? count : ch_queue_count(ch_dict_get(moves_history, sessionId)));

                            player = player == 0 ? 1 : 0;

                            FREE(board2);
                            FREE(board_next2);
                        }
                    }

                    FREE(board1);
                    FREE(board_next1);
                }
				FREE(mfen_next);
            }

            if (sfenn != NULL) FREE(sfenn);
            if (mfenn != NULL) FREE(mfenn);
            if (moves != NULL) FREE(moves);
            if (move != NULL) FREE(move);
            if (fen != NULL) FREE(fen);
            if (print) ch_print_board(valid_fen);
        }
        else if (strncmp(buff, "go", 2) == 0 || strncmp(buff, "go infinite", 11) == 0) {

            int qcount = ch_queue_count(ch_dict_get(moves_history,sessionId));

            player = qcount % 2 == 0 ? 0 : 1;

            move_state = ch_self_learn_step(sessionId, sfen, level, nets, valid_fen, valid_fen_move, 1);

            if (++nets->netv->nsteps % 1000 == 0) save_weights(nets->netv, nets->netv_wname);
            if (++nets->netp->nsteps % 1000 == 0) save_weights(nets->netp, nets->netp_wname);

            strcpy(valid_fen, move_state.fen);
            strcpy(valid_fen_move, move_state.move);

            fprintf(stdout, "info depth %i pv %s\n", 1, valid_fen_move);

            if (strcmp(valid_fen_move, "") == 0) {
                fprintf(stdout, "bestmove\n");
            } else {
                fprintf(stdout, "bestmove %s ponder %s\n", valid_fen_move, valid_fen_move);
            }

            fflush(stdout);
            if (print) ch_print_board(valid_fen);
        }
        else if (strncmp(buff, "register", 8) == 0) {
            fprintf(stdout,"registration ok\n");
            fflush(stdout);
        }

        FREE(buff);
        buff = NULL;
    }

    save_weights(nets->netv, nets->netv_wname);
    save_weights(nets->netp, nets->netp_wname);

    ch_free_networks(nets);

    ch_clean_history(sessionId, 1);

    if (print) fclose(log);
}

void test_mchess(int argc, char** argv, char *cfgfile, char *weight_file) {
    char valid_fen[128];
    char valid_fen_move[8];
    char sfen[128];
    char valid_fen_next[128];
    char valid_fen_last[128];

    ch_board_state move_state = {0};

    int print = 0;

    char* startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    int level = 0;

    ch_nets* nets = ch_load_networks(cfgfile, weight_file);

    char* sessionId = "13a25e80-ece3-4a4b-9347-e6df74386d02";

    ch_init_game_history(sessionId);

    FILE* log = NULL;
    if (print) log = fopen("log.txt", "a+");

    fprintf(stdout, "iChess.io by Piotr Sowa v7.27\n");

    strcpy(valid_fen, startpos);
    strcpy(valid_fen_move, "");
    //ch_get_fen_960(valid_fen);
    strcpy(sfen, valid_fen);

    int player = 0;
    ch_clean_history(sessionId, 1);

    FILE *sf = popen("/usr/local/Cellar/stockfish/17.1/bin/stockfish", "r+");
    if (!sf) {
        fprintf(stderr, "Failed to start Stockfish\n");
        exit(1);
    }

    fprintf(sf, "uci\n"); fflush(sf);

    char *buff = NULL;
    size_t len = 0;
    size_t nread = 0;
    while ((nread = getline(&buff, &len, stdin)) > 0) {
        if (buff == NULL) continue;
        buff[strcspn(buff, "\r\n")] = 0;

        if (print) {
            fprintf(log, "%s\n", buff);
            fflush(log);
        }
        if (strncmp(buff, "ucinewgame", 10) == 0) {
            strcpy(move_state.fen, valid_fen);
            strcpy(move_state.move, valid_fen_move);
            strcpy(valid_fen, sfen[0] != '\0' ? sfen : startpos);
            strcpy(valid_fen_move, "");
            strcpy(sfen, "");
            player = 0;
            ch_clean_history(sessionId, 1);
			save_weights(nets->netv, nets->netv_wname);
            save_weights(nets->netp, nets->netp_wname);
            fprintf(stdout, "%s\n", "uciok");
            fflush(stdout);
            fprintf(sf, "%s\n", buff); fflush(sf);
        }
        else if (strncmp(buff, "uci", 3) == 0) {
            strcpy(valid_fen, startpos);
            strcpy(valid_fen_move, "");
            fprintf(stdout, "%s\n", "id name iChess.io 7.27");
            fprintf(stdout, "%s\n", "id author Piotr Sowa");
            fprintf(stdout, "%s\n", "option name UCI_Chess960 type check default false");
            fprintf(stdout, "%s\n", "option name BackendOptions type string default");
            fprintf(stdout, "%s\n", "option name Ponder type check default false");
            fprintf(stdout, "%s\n", "option name MultiPV type spin default 1 min 1 max 500");
            fprintf(stdout, "%s\n", "uciok");
            fflush(stdout);
            fprintf(sf, "%s\n", buff); fflush(sf);
        }
        else if (strncmp(buff, "isready", 7) == 0) {
            fprintf(stdout,"readyok\n");
            fflush(stdout);
            fprintf(sf, "%s\n", buff); fflush(sf);
        }
        else if (strncmp(buff, "stop", 4) == 0) {
            fprintf(sf, "%s\n", buff); fflush(sf);
        }
        else if (strncmp(buff, "quit", 4) == 0) {
            fprintf(sf, "%s\n", buff); fflush(sf);
            break;
        }
        else if (strncmp(buff, "position ", 9) == 0){
            fprintf(sf, "%s\n", buff); fflush(sf);

            char** moves = NULL;
            char* move = NULL;
            char* sfenn = NULL;
            char* mfenn = NULL;

            int count = 0;
            char* fen = ch_analyze_pos(sfen, buff, &sfenn, &mfenn, &moves, &move, &count);

            strcpy(sfen, sfenn != NULL ? sfenn : "");
            strcpy(valid_fen, fen != NULL ? fen : "");
            strcpy(valid_fen_move, move != NULL ? move : "");
            strcpy(valid_fen_next, sfenn != NULL ? sfenn : "");
            strcpy(valid_fen_last, mfenn != NULL ? mfenn : "");

            char *mfen_next = NULL;
            int mfen_next_idx = 0;
            int mfen_next_cnt = 0;

            int qcount = ch_queue_count(ch_dict_get(moves_history,sessionId));
            int exists = qcount > 1;

            player = qcount % 2 == 0 ? 1 : 0;

            int addon = (exists ? count : qcount);

            if (exists) {

                if (ch_board_after_move(sfen, valid_fen_last, moves[count - 1], &mfen_next, &mfen_next_idx, &mfen_next_cnt)) {

                    strcpy(move_state.fen, valid_fen_last);
                    strcpy(move_state.move, moves[count - 1]);
                    move_state.indext = mfen_next_idx;

                    //ch_put_into_game_queue(sessionId, valid_fen_last, moves[count - 1], mfen_next_idx, nets, sfen);

                    float *board0 = ch_fen_to_board(valid_fen_last, 1);
                    float *board_next0 = ch_fen_to_board(mfen_next, 1);
                    float pow0 = ch_eval_the_move(sfen, valid_fen_last, mfen_next);
                    float powW0 = 0;
                    float powB0 = 0;
                    ch_eval_the_board(sfen, board0, &powW0, &powB0);
                    float value0 = player == 0 ? powW0 : powB0;

                    fprintf(stderr,
                            "pick: step %ld(%s): count: (%i) checked: (%i) power: (%.7f) index: (%i)\n",
                            nets->netv->nsteps, player == 0 ? "w" : "b", mfen_next_cnt, 1, pow0, mfen_next_idx);

                    ch_self_study_train_self_step(sessionId, sfen, valid_fen_last, moves[count - 1], nets, level, mfen_next_idx, pow0, value0);

                    FREE(board0);
                    FREE(board_next0);

                }
				FREE(mfen_next);
            }
            else if (count > 0) {

                if (ch_board_after_move(sfen, valid_fen_next, moves[addon - 1], &mfen_next, &mfen_next_idx, &mfen_next_cnt)) {

                    strcpy(move_state.fen, valid_fen_next);
                    strcpy(move_state.move, moves[addon - 1]);
                    move_state.indext = mfen_next_idx;

                    ch_put_into_game_queue(sessionId, valid_fen_next, moves[addon - 1], mfen_next_idx, nets, sfen);

                    if (++nets->netv->nsteps % 1000 == 0) save_weights(nets->netv, nets->netv_wname);
                    if (++nets->netp->nsteps % 1000 == 0) save_weights(nets->netp, nets->netp_wname);

                    float *board1 = ch_fen_to_board(valid_fen_next, 1);
                    float *board_next1 = ch_fen_to_board(mfen_next, 1);
                    float pow1 = ch_eval_the_move(sfen, valid_fen_next, mfen_next);
                    float powW1 = 0;
                    float powB1 = 0;
                    ch_eval_the_board(sfen, board1, &powW1, &powB1);
                    float value1 = player == 0 ? powW1 : powB1;

                    fprintf(stderr,
                            "pick: step %ld(%s): count: (%i) checked: (%i) power: (%.7f) index: (%i)\n",
                            nets->netv->nsteps, player == 0 ? "w" : "b", mfen_next_cnt, 1, pow1, mfen_next_idx);

                    ch_self_study_train_self_step(sessionId, sfen, valid_fen_next, moves[addon - 1], nets, level, mfen_next_idx, pow1, value1);

                    strcpy(valid_fen_next, mfen_next);
                    strcpy(valid_fen_last, mfen_next);

                    strcpy(valid_fen, valid_fen_next);
                    strcpy(valid_fen_move, moves[addon - 1]);

                    addon = (exists ? count : ch_queue_count(ch_dict_get(moves_history, sessionId)));

                    for (int i = addon - 1; i < count; ++i) {

                        if (ch_board_after_move(sfen, valid_fen_next, moves[addon - 1], &mfen_next, &mfen_next_idx, &mfen_next_cnt)) {

                            strcpy(move_state.fen, valid_fen_next);
                            strcpy(move_state.move, moves[addon - 1]);
                            move_state.indext = mfen_next_idx;

                            ch_put_into_game_queue(sessionId, valid_fen_next, moves[addon - 1], mfen_next_idx, nets , sfen);

                            if (++nets->netv->nsteps % 1000 == 0) save_weights(nets->netv, nets->netv_wname);
                            if (++nets->netp->nsteps % 1000 == 0) save_weights(nets->netp, nets->netp_wname);

                            float *board2 = ch_fen_to_board(valid_fen_next, 1);
                            float *board_next2 = ch_fen_to_board(mfen_next, 1);
                            float pow2 = ch_eval_the_move(sfen, valid_fen_next, mfen_next);
                            float powW2 = 0;
                            float powB2 = 0;
                            ch_eval_the_board(sfen, board2, &powW2, &powB2);
                            float value2 = player == 0 ? powW2 : powB2;

                            fprintf(stderr,
                                    "pick: step %ld(%s): count: (%i) checked: (%i) power: (%.7f) index: (%i)\n",
                                    nets->netv->nsteps, player == 0 ? "w" : "b", mfen_next_cnt, 1, pow2, mfen_next_idx);

                            ch_self_study_train_self_step(sessionId, sfen, valid_fen_next, moves[addon - 1], nets, level,mfen_next_idx, pow2, value2);

                            strcpy(valid_fen_next, mfen_next);
                            strcpy(valid_fen_last, mfen_next);

                            strcpy(valid_fen, valid_fen_next);
                            strcpy(valid_fen_move, moves[addon - 1]);

                            addon = (exists ? count : ch_queue_count(ch_dict_get(moves_history, sessionId)));

                            player = player == 0 ? 1 : 0;

                            FREE(board2);
                            FREE(board_next2);
                        }
                    }

                    FREE(board1);
                    FREE(board_next1);
                }
				FREE(mfen_next);
            }

            if (sfenn != NULL) FREE(sfenn);
            if (mfenn != NULL) FREE(mfenn);
            if (moves != NULL) FREE(moves);
            if (move != NULL) FREE(move);
            if (fen != NULL) FREE(fen);
            if (print) ch_print_board(valid_fen);
        }
        else if (strncmp(buff, "go", 2) == 0 || strncmp(buff, "go infinite", 11) == 0) {
            fprintf(sf, "%s\n", buff); fflush(sf);

            char sf_line[256];
            char sf_move[16] = "";
            while (fgets(sf_line, sizeof(sf_line), sf)) {
                if (strncmp(sf_line, "bestmove", 8) == 0) {
                    sscanf(sf_line, "bestmove %s", sf_move);
                    break;
                }
            }

            if (strcmp(sf_move, "") != 0) {
                strcpy(valid_fen_move, sf_move);
            }

            int qcount = ch_queue_count(ch_dict_get(moves_history,sessionId));

            player = qcount % 2 == 0 ? 0 : 1;

            strcpy(move_state.fen, valid_fen);
            strcpy(move_state.move, valid_fen_move);

            fprintf(stderr,
                    "pick: step %ld(%s): count: (%i) checked: (%i)\n",
                    nets->netv->nsteps, player == 0 ? "w" : "b", 0, 1);
            ch_self_learn_step(sessionId, sfen, level, nets, valid_fen, valid_fen_move, 1);

            if (++nets->netv->nsteps % 1000 == 0) save_weights(nets->netv, nets->netv_wname);
            if (++nets->netp->nsteps % 1000 == 0) save_weights(nets->netp, nets->netp_wname);

            strcpy(valid_fen, move_state.fen);
            strcpy(valid_fen_move, move_state.move);

            fprintf(stdout, "info depth %i pv %s\n", 1, valid_fen_move);

            if (strcmp(valid_fen_move, "") == 0) {
                fprintf(stdout, "bestmove\n");
            } else {
                fprintf(stdout, "bestmove %s ponder %s\n", valid_fen_move, valid_fen_move);
            }

            fflush(stdout);
            if (print) ch_print_board(valid_fen);
        }
        else if (strncmp(buff, "register", 8) == 0) {
            fprintf(sf, "%s\n", buff); fflush(sf);

            fprintf(stdout,"registration ok\n");
            fflush(stdout);
        }

        FREE(buff);
        buff = NULL;
    }

    save_weights(nets->netv, nets->netv_wname);
    save_weights(nets->netp, nets->netp_wname);

    ch_free_networks(nets);

    ch_clean_history(sessionId, 1);

    pclose(sf);

    if (print) fclose(log);
}

static float schess_stockfish_eval_cp_norm_depth20(const char *fen) {
    const char *sf_path = "/usr/local/Cellar/stockfish/17.1/bin/stockfish";
    FILE *sf = popen(sf_path, "r+");
    if (!sf) return 0.0f;

    fprintf(sf, "uci\n"); fflush(sf);
    char line[512];
    int ok = 0;
    for (int i = 0; i < 2000 && fgets(line, sizeof(line), sf); ++i) { if (strstr(line, "uciok")) { ok = 1; break; } }
    if (!ok) { pclose(sf); return 0.0f; }

    fprintf(sf, "setoption name UCI_Chess960 value true\n"); fflush(sf);
    fprintf(sf, "isready\n"); fflush(sf);
    ok = 0;
    for (int i = 0; i < 2000 && fgets(line, sizeof(line), sf); ++i) { if (strstr(line, "readyok")) { ok = 1; break; } }
    if (!ok) { pclose(sf); return 0.0f; }

    fprintf(sf, "position fen %s\n", fen); fflush(sf);
    fprintf(sf, "go depth 20\n"); fflush(sf);

    float cp = 0.0f;
    int have_cp = 0, have_mate = 0, mate = 0;
    while (fgets(line, sizeof(line), sf)) {
        const char *s = strstr(line, "score ");
        if (s) {
            s += 6;
            if (!strncmp(s, "cp ", 3)) { s += 3; int v=0; if (sscanf(s, "%d", &v)==1){ cp=(float)v; have_cp=1; have_mate=0; } }
            else if (!strncmp(s, "mate ", 5)) { s += 5; int m=0; if (sscanf(s, "%d", &m)==1){ mate=m; have_mate=1; } }
        }
        if (!strncmp(line, "bestmove", 8)) break;
    }
    pclose(sf);

    if (have_mate) return (mate>0)? 1.0f : -1.0f;
    if (have_cp) {
        float v = cp / 1000.0f;
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        return v;
    }
    return 0.0f;
}

void test_schess(int argc, char** argv, char *cfgfile, char *weight_file) {
    char valid_fen[256];
    char valid_fen_move[32];
    char sfen[256];
    char valid_fen_next[256];
    char valid_fen_last[256];

    ch_board_state move_state = (ch_board_state){0};

    int level = 0;

    ch_nets* nets = ch_load_networks(cfgfile, weight_file);
    set_batch_network(nets->netp, 1);
    set_batch_network(nets->netv, 1);

    char* startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    strcpy(valid_fen, startpos);
    strcpy(valid_fen_move, "");
    strcpy(sfen, valid_fen);

    char sessionId[64]; snprintf(sessionId, sizeof(sessionId), "%s", "schess-session");
    ch_init_game_history(sessionId);
    ch_clean_history(sessionId, 1);

    printf("id name iChess.io\n");
    printf("id author iSowa.io\n");
    printf("uciok\n");
    fflush(stdout);

    char *buff = NULL;
    size_t len = 0;
    int step_counter = 0;

    while (1) {
        ssize_t nread = getline(&buff, &len, stdin);
        if (nread <= 0) break;
        buff[strcspn(buff, "\r\n")] = 0;
        if (!buff[0]) continue;

        if (!strcmp(buff, "quit")) break;

        if (!strcmp(buff, "uci")) {
            printf("id name iChess.io\n");
            printf("id author iSowa.io\n");
            printf("uciok\n");
            fflush(stdout);
            continue;
        }

        if (!strcmp(buff, "isready")) {
            printf("readyok\n");
            fflush(stdout);
            continue;
        }

        if (!strcmp(buff, "ucinewgame")) {
            ch_clean_history(sessionId, 1);
            strcpy(valid_fen, startpos);
            strcpy(sfen, valid_fen);
            valid_fen_move[0]=0;
            continue;
        }

        if (!strncmp(buff, "setoption", 9)) {
            // accept silently
            continue;
        }

        if (!strncmp(buff, "position ", 9)) {
            char *sfenn = NULL, *mfenn = NULL, **moves = NULL, *move = NULL;
            int count = 0;
            ch_analyze_pos(sfen, buff, &sfenn, &mfenn, &moves, &move, &count);

            strcpy(sfen, sfenn != NULL ? sfenn : "");
            strcpy(valid_fen_last, (mfenn && mfenn[0]) ? mfenn : sfen);

            if (move && move[0]) {
                char *fen_next = NULL;
                int fen_next_idx = -1, fen_next_cnt = 0;
                if (ch_board_after_move(sfen, valid_fen_last, move, &fen_next, &fen_next_idx, &fen_next_cnt)) {
                    float value_eval = schess_stockfish_eval_cp_norm_depth20(fen_next ? fen_next : valid_fen_last);
                    float pow_eval   = ch_eval_the_move(sfen, valid_fen_last, move);

                    ch_self_study_train_self_step(sessionId, sfen, valid_fen_last, move, nets, level, fen_next_idx, pow_eval, value_eval);

                    strcpy(valid_fen, fen_next ? fen_next : valid_fen_last);
                    if (fen_next) FREE(fen_next);
                }
            }

            if (moves) { for (int i=0;i<count;++i) if (moves[i]) FREE(moves[i]); FREE(moves); }
            if (sfenn) FREE(sfenn);
            if (mfenn) FREE(mfenn);
            if (move) FREE(move);
            continue;
        }

        if (!strncmp(buff, "go", 2)) {
            char *valids_fen = NULL;
            char **valid_moves = NULL;
            int valid_moves_count = 0;
            if (!ch_get_all_valid_moves(sfen, valid_fen, &valids_fen, &valid_moves, &valid_moves_count) || valid_moves_count <= 0) {
                printf("bestmove 0000\n");
                fflush(stdout);
                if (valids_fen) FREE(valids_fen);
                continue;
            }

            int best_idx = -1;
            float best_score = -FLT_MAX;
            for (int i = 0; i < valid_moves_count; ++i) {
                float sc = ch_eval_the_move(sfen, valids_fen, valid_moves[i]);
                if (sc > best_score) { best_score = sc; best_idx = i; }
            }

            const char *bm = (best_idx >= 0) ? valid_moves[best_idx] : "0000";
            printf("bestmove %s\n", bm);
            fflush(stdout);

            if (best_idx >= 0) {
                char *fen_next = NULL;
                int fen_next_idx = -1, fen_next_cnt = 0;
                if (ch_board_after_move(sfen, valids_fen, valid_moves[best_idx], &fen_next, &fen_next_idx, &fen_next_cnt)) {
                    strcpy(valid_fen, fen_next ? fen_next : valids_fen);
                    if (fen_next) FREE(fen_next);
                }
            }

            for (int i = 0; i < valid_moves_count; ++i) if (valid_moves[i]) FREE(valid_moves[i]);
            if (valid_moves) FREE(valid_moves);
            if (valids_fen) FREE(valids_fen);

            if ((++step_counter % 1000) == 0) {
                save_weights(nets->netv, nets->netv_wname);
                save_weights(nets->netp, nets->netp_wname);
            }
            continue;
        }

        // ignore others
    }

    if (buff) { free(buff); buff = NULL; }

    save_weights(nets->netv, nets->netv_wname);
    save_weights(nets->netp, nets->netp_wname);

    ch_free_networks(nets);
    ch_clean_history(sessionId, 1);
}
