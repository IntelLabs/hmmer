#ifndef _FWDBACK_AVX512_HPP
#define _FWDBACK_AVX512_HPP

#include "easel.h"
#include "hmmer.h"
#include "p7_config.h"

#ifdef __cplusplus
extern "C" {
#endif

int forward_engine_avx512(const int do_full, const ESL_DSQ *dsq, const int L, const P7_OPROFILE *om, P7_OMX *ox, float *opt_sc);
int forward_engine_old(const int do_full, const ESL_DSQ *dsq, const int L, const P7_OPROFILE *om, P7_OMX *ox, float *opt_sc);

#ifdef __cplusplus
} //extern "C"
#endif

#endif