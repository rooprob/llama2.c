/*
Convert binary model to big endian

Example compile: (see README for more details)
$ gcc -O3 -o convert convert.c

Then run with:
$ ./convert model.bin bemodel.bin
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
// ----------------------------------------------------------------------------
int ibyteswap(int i) {
    union {
        int i;
        char b[4];
    } src, dst;

    src.i = i;
    dst.b[3] = src.b[0];
    dst.b[2] = src.b[1];
    dst.b[1] = src.b[2];
    dst.b[0] = src.b[3];
    return dst.i;
}
// Transformer and RunState structs, and related memory management

typedef struct {
    int dim;         // transformer dimension
    int hidden_dim;  // for ffn layers
    int n_layers;    // number of layers
    int n_heads;     // number of query heads
    int n_kv_heads;  // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size;  // vocabulary size, usually 256 (byte-level)
    int seq_len;     // max sequence length
} Config;

void convert_config(Config* p, Config* c) {
    c->dim = ibyteswap(p->dim);
    c->hidden_dim = ibyteswap(p->hidden_dim);
    c->n_layers = ibyteswap(p->n_layers);
    c->n_heads = ibyteswap(p->n_heads);
    c->n_kv_heads = ibyteswap(p->n_kv_heads);
    c->vocab_size = ibyteswap(p->vocab_size);
    c->seq_len = ibyteswap(p->seq_len);
}

void print_config(Config* p) {
    printf("dim: %d\n", p->dim);
    printf("hidden_dim: %d\n", p->hidden_dim);
    printf("n_layers: %d\n", p->n_layers);
    printf("n_heads: %d\n", p->n_heads);
    printf("n_kv_heads: %d\n", p->n_kv_heads);
    printf("vocab_size: %d\n", p->vocab_size);
    printf("seq_len: %d\n", p->seq_len);
}


typedef struct {
    // token embedding table
    float* token_embedding_table;  // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight;  // (layer, dim) rmsnorm weights
    float* rms_ffn_weight;  // (layer, dim)
    // weights for matmuls
    float* wq;  // (layer, dim, dim)
    float* wk;  // (layer, dim, dim)
    float* wv;  // (layer, dim, dim)
    float* wo;  // (layer, dim, dim)
    // weights for ffn
    float* w1;  // (layer, hidden_dim, dim)
    float* w2;  // (layer, dim, hidden_dim)
    float* w3;  // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight;  // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    float* freq_cis_real;  // (seq_len, dim/2)
    float* freq_cis_imag;  // (seq_len, dim/2)
} TransformerWeights;

void malloc_weights(TransformerWeights* w, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    w->token_embedding_table = calloc(p->vocab_size * p->dim, sizeof(float));
    w->rms_att_weight = calloc(p->n_layers * p->dim, sizeof(float));
    w->rms_ffn_weight = calloc(p->n_layers * p->dim, sizeof(float));
    w->wq = calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    w->wk = calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    w->wv = calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    w->wo = calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    w->w1 = calloc(p->n_layers * p->hidden_dim * p->dim, sizeof(float));
    w->w2 = calloc(p->n_layers * p->dim * p->hidden_dim, sizeof(float));
    w->w3 = calloc(p->n_layers * p->hidden_dim * p->dim, sizeof(float));
    w->rms_final_weight = calloc(p->dim, sizeof(float));
    w->freq_cis_real = calloc(p->seq_len * p->dim / 2, sizeof(float));
    w->freq_cis_imag = calloc(p->seq_len * p->dim / 2, sizeof(float));
    // ensure all mallocs went fine
    if (!w->token_embedding_table || !w->rms_att_weight || !w->rms_ffn_weight || !w->wq || !w->wk || !w->wv || !w->wo || !w->w1 || !w->w2 || !w->w3 ||
        !w->rms_final_weight || !w->freq_cis_real || !w->freq_cis_imag) {
        printf("malloc failed!\n");
        exit(1);
    }
}

void free_weights(TransformerWeights* w) {
    free(w->token_embedding_table);
    free(w->rms_att_weight);
    free(w->rms_ffn_weight);
    free(w->wq);
    free(w->wk);
    free(w->wv);
    free(w->wo);
    free(w->w1);
    free(w->w2);
    free(w->w3);
    free(w->rms_final_weight);
    free(w->freq_cis_real);
    free(w->freq_cis_imag);
}

// ----------------------------------------------------------------------------
// initialization: read from checkpoint
float fbyteswap(float f) {
    union {
        float f;
        char b[4];
    } src, dst;

    src.f = f;
    dst.b[3] = src.b[0];
    dst.b[2] = src.b[1];
    dst.b[1] = src.b[2];
    dst.b[0] = src.b[3];
    return dst.f;
}

float* fabyteswap(float* f, int size) {
    for (int idx = 0; idx < size; idx++) {
        f[idx] = fbyteswap(f[idx]);
    }

    return f;
}

int convert_weights(TransformerWeights* w, Config* p, FILE* f, FILE* c) {
    if (fread(w->token_embedding_table, sizeof(float), p->vocab_size * p->dim, f) != p->vocab_size * p->dim) return 1;
    fabyteswap(w->token_embedding_table, p->vocab_size * p->dim);
    if (fwrite(w->token_embedding_table, sizeof(float), p->vocab_size * p->dim, c) != p->vocab_size * p->dim) return 1;

    if (fread(w->rms_att_weight, sizeof(float), p->n_layers * p->dim, f) != p->n_layers * p->dim) return 1;
    fabyteswap(w->rms_att_weight, p->n_layers * p->dim);
    if (fwrite(w->rms_att_weight, sizeof(float), p->n_layers * p->dim, c) != p->n_layers * p->dim) return 1;

    if (fread(w->wq, sizeof(float), p->n_layers * p->dim * p->dim, f) != p->n_layers * p->dim * p->dim) return 1;
    fabyteswap(w->wq, p->n_layers * p->dim * p->dim);
    if (fwrite(w->wq, sizeof(float), p->n_layers * p->dim * p->dim, c) != p->n_layers * p->dim * p->dim) return 1;

    if (fread(w->wk, sizeof(float), p->n_layers * p->dim * p->dim, f) != p->n_layers * p->dim * p->dim) return 1;
    fabyteswap(w->wk, p->n_layers * p->dim * p->dim);
    if (fwrite(w->wk, sizeof(float), p->n_layers * p->dim * p->dim, c) != p->n_layers * p->dim * p->dim) return 1;

    if (fread(w->wv, sizeof(float), p->n_layers * p->dim * p->dim, f) != p->n_layers * p->dim * p->dim) return 1;
    fabyteswap(w->wv, p->n_layers * p->dim * p->dim);
    if (fwrite(w->wv, sizeof(float), p->n_layers * p->dim * p->dim, c) != p->n_layers * p->dim * p->dim) return 1;

    if (fread(w->wo, sizeof(float), p->n_layers * p->dim * p->dim, f) != p->n_layers * p->dim * p->dim) return 1;
    fabyteswap(w->wo, p->n_layers * p->dim * p->dim);
    if (fwrite(w->wo, sizeof(float), p->n_layers * p->dim * p->dim, c) != p->n_layers * p->dim * p->dim) return 1;

    if (fread(w->rms_ffn_weight, sizeof(float), p->n_layers * p->dim, f) != p->n_layers * p->dim) return 1;
    fabyteswap(w->rms_ffn_weight, p->n_layers * p->dim);
    if (fwrite(w->rms_ffn_weight, sizeof(float), p->n_layers * p->dim, c) != p->n_layers * p->dim) return 1;

    if (fread(w->w1, sizeof(float), p->n_layers * p->dim * p->hidden_dim, f) != p->n_layers * p->dim * p->hidden_dim) return 1;
    fabyteswap(w->w1, p->n_layers * p->dim * p->hidden_dim);
    if (fwrite(w->w1, sizeof(float), p->n_layers * p->dim * p->hidden_dim, c) != p->n_layers * p->dim * p->hidden_dim) return 1;

    if (fread(w->w2, sizeof(float), p->n_layers * p->hidden_dim * p->dim, f) != p->n_layers * p->hidden_dim * p->dim) return 1;
    fabyteswap(w->w2, p->n_layers * p->hidden_dim * p->dim);
    if (fwrite(w->w2, sizeof(float), p->n_layers * p->hidden_dim * p->dim, c) != p->n_layers * p->hidden_dim * p->dim) return 1;

    if (fread(w->w3, sizeof(float), p->n_layers * p->dim * p->hidden_dim, f) != p->n_layers * p->dim * p->hidden_dim) return 1;
    fabyteswap(w->w3, p->n_layers * p->dim * p->hidden_dim);
    if (fwrite(w->w3, sizeof(float), p->n_layers * p->dim * p->hidden_dim, c) != p->n_layers * p->dim * p->hidden_dim) return 1;

    if (fread(w->rms_final_weight, sizeof(float), p->dim, f) != p->dim) return 1;
    fabyteswap(w->rms_final_weight, p->dim);
    if (fwrite(w->rms_final_weight, sizeof(float), p->dim, c) != p->dim) return 1;

    int head_size = p->dim / p->n_heads;
    if (fread(w->freq_cis_real, sizeof(float), p->seq_len * head_size / 2, f) != p->seq_len * head_size / 2) return 1;
    fabyteswap(w->freq_cis_real, p->seq_len * head_size / 2);
    if (fwrite(w->freq_cis_real, sizeof(float), p->seq_len * head_size / 2, c) != p->seq_len * head_size / 2) return 1;

    if (fread(w->freq_cis_imag, sizeof(float), p->seq_len * head_size / 2, f) != p->seq_len * head_size / 2) return 1;
    fabyteswap(w->freq_cis_imag, p->seq_len * head_size / 2);
    if (fwrite(w->freq_cis_imag, sizeof(float), p->seq_len * head_size / 2, c) != p->seq_len * head_size / 2) return 1;

    return 0;
}

// ----------------------------------------------------------------------------

long time_in_ms() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return time.tv_sec * 1000 + time.tv_usec / 1000;
}

int main(int argc, char* argv[]) {
    // poor man's C argparse
    char* checkpoint = NULL;  // e.g. out/model.bin
    char* converted = NULL;   // e.g. out/bemodel.out
    // 'checkpoint' is necessary arg
    if (argc != 3) {
        printf("Usage: %s <checkpoint_file> <converted_file>\n", argv[0]);
        return 1;
    }
    checkpoint = argv[1];
    converted = argv[2];

    // the current position we are in
    long start = time_in_ms();

    // read in the model.bin file
    Config cconfig;
    Config config;
    TransformerWeights weights;
    {
        FILE* file = fopen(checkpoint, "rb");
        if (!file) {
            printf("Unable to open the checkpoint file %s!\n", checkpoint);
            return 1;
        }
        // rewrite_converted_file
        FILE* cfile = fopen(converted, "wb");
        if (!cfile) {
            printf("Unable to open the converted file %s!\n", converted);
            return 1;
        }

        // read in the config header
        if (fread(&config, sizeof(Config), 1, file) != 1) {
            return 1;
        }
        // convert Config to network byte order
        convert_config(&config, &cconfig);
        // write the converted config header
        if (fwrite(&cconfig, sizeof(Config), 1, cfile) != 1) {
            printf("error: converting config\n");
            return 1;
        }
        // read in the Transformer weights
        malloc_weights(&weights, &cconfig);
        if (convert_weights(&weights, &cconfig, file, cfile)) {
            printf("error: converting weights\n");
            return 1;
        }
        fclose(file);
        fclose(cfile);
    }

    // report achieved tok/s
    long end = time_in_ms();
    printf("\nconverted: %fms\n", (double)(end - start));

    // memory cleanup
    free_weights(&weights);
    return 0;
}
