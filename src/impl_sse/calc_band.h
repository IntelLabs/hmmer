#ifndef _CALC_BAND_HPP
#define _CALC_BAND_HPP

#include "easel.h"
#include "hmmer.h"
#include "p7_config.h"
#include <emmintrin.h> /* SSE2 */

#ifdef __cplusplus
extern "C" {
#endif

__m128i calc_band_avx512(const ESL_DSQ *, const int, const P7_OPROFILE *,
                                  const int, const __m128i, __m128i, const int);

void print_m128i(__m128i);

void print_m512i(__m512i);
void print_128_num(__m128i var);
#ifdef __cplusplus
} //extern "C"
#endif

#endif