#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PpDocEngine PpDocEngine;

PpDocEngine * ppdoc_create(void);
void ppdoc_destroy(PpDocEngine * engine);

bool ppdoc_load_manifest(PpDocEngine * engine, const char * path);
bool ppdoc_load_weights(PpDocEngine * engine, const char * path);

bool ppdoc_prepare_plan_prefix(
    PpDocEngine * engine,
    const char * plan_path,
    const char * output_name);

void ppdoc_clear_plan_prefix(PpDocEngine * engine);

// output_elems is an in/out parameter: pass the output buffer capacity on input;
// it is set to the required element count before the function returns.
bool ppdoc_run_plan_prefix(
    PpDocEngine * engine,
    const char * plan_path,
    const char * output_name,
    const float * input,
    int64_t input_elems,
    float * output,
    int64_t * output_elems,
    int64_t out_shape_nchw[4]);

bool ppdoc_run_prepared_plan_prefix(
    PpDocEngine * engine,
    const float * input,
    int64_t input_elems,
    float * output,
    int64_t * output_elems,
    int64_t out_shape_nchw[4]);

const char * ppdoc_last_error(PpDocEngine * engine);

#ifdef __cplusplus
}
#endif
