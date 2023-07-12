#include <immintrin.h>
#include "hmmer.h"
#include "p7_config.h"
#include "esl_sse.h"
#include "fwdback_avx512.h"
//#include <vector>
//#include <memory>
//#include <iostream>
//#include <cstring>
#include <assert.h>

//template <typename T>
//void CoutM128(__m128 m)
//{
//  T *m_ = (T *)&m;
//  assert(16 % sizeof(T) == 0);
//  size_t N = 16 / sizeof(T);
//  for (size_t n = 0; n < N; ++n)
//  {
//    std::cout << m_[n] << ", ";
//  }
//  std::cout << std::endl;
//}
//
//template <typename T>
//void CoutM512(__m512 m)
//{
//  T *m_ = (T *)&m;
//  assert(64 % sizeof(T) == 0);
//  size_t N = 64 / sizeof(T);
//  for (size_t n = 0; n < N; ++n)
//  {
//    std::cout << m_[n] << ", ";
//  }
//  std::cout << std::endl;
//}
static const __mmask16 mask = 0x000F;

//static inline __m512 esl_sse_rightshift_ps_512(__m512 a, __m512 b)
//{
//  auto tmp = _mm512_shuffle_ps(a, a, _MM_SHUFFLE(2, 1, 0, 0));
//  return _mm512_mask_mov_ps(tmp, mask, b);
//}

//struct free_delete
//{
//  void operator()(void *x) { free(x); }
//};

int forward_engine_avx512(const int do_full, const ESL_DSQ *dsq, const int L, const P7_OPROFILE *om, P7_OMX *ox, float *opt_sc)
{
  register __m128 mpv, dpv, ipv;
  register __m128 sv;
  register __m128 dcv;
  register __m128 xEv;
  register __m128 xBv;
  const __m128 zerov = _mm_setzero_ps();
  float xN, xE, xB, xC, xJ;
  int i;
  int q;
  int j;
  const int Q = ((((2) > (((((om->M) - 1) / 4) + 1))) ? (2) : (((((om->M) - 1) / 4) + 1))));
  __m128 *dpc = ox->dpf[0];
  __m128 *dpp;
  __m128 *rp;
  __m128 *tp;

  ox->M = om->M;
  ox->L = L;
  ox->has_own_scales = 1;
  for (q = 0; q < Q; q++)
    ((dpc)[(q)*3 + p7X_M]) = ((dpc)[(q)*3 + p7X_I]) = ((dpc)[(q)*3 + p7X_D]) = zerov;
  xE = ox->xmx[p7X_E] = 0.;
  xN = ox->xmx[p7X_N] = 1.;
  xJ = ox->xmx[p7X_J] = 0.;
  xB = ox->xmx[p7X_B] = om->xf[p7O_N][p7O_MOVE];
  xC = ox->xmx[p7X_C] = 0.;

  ox->xmx[p7X_SCALE] = 1.0;
  ox->totscale = 0.0;

  for (i = 1; i <= L; i++)
  {
    dpp = dpc;
    dpc = ox->dpf[do_full * i];
    rp = om->rfv[dsq[i]];
    tp = om->tfv;
    dcv = _mm_setzero_ps();
    xEv = _mm_setzero_ps();
    xBv = _mm_set1_ps(xB);

    mpv = esl_sse_rightshift_ps(((dpp)[(Q - 1) * 3 + p7X_M]), zerov);
    dpv = esl_sse_rightshift_ps(((dpp)[(Q - 1) * 3 + p7X_D]), zerov);
    ipv = esl_sse_rightshift_ps(((dpp)[(Q - 1) * 3 + p7X_I]), zerov);

    for (q = 0; q < Q; q++)
    {

      sv = _mm_mul_ps(xBv, *tp++);
      sv = _mm_fmadd_ps(mpv, *tp++, sv);
      sv = _mm_fmadd_ps(ipv, *tp++, sv);
      sv = _mm_fmadd_ps(dpv, *tp++, sv);
      sv = _mm_mul_ps(sv, *rp++);
      xEv = _mm_add_ps(xEv, sv);

      mpv = ((dpp)[(q)*3 + p7X_M]);
      dpv = ((dpp)[(q)*3 + p7X_D]);
      ipv = ((dpp)[(q)*3 + p7X_I]);

      ((dpc)[(q)*3 + p7X_M]) = sv;
      ((dpc)[(q)*3 + p7X_D]) = dcv;

      dcv = _mm_mul_ps(sv, *tp++);

      sv = _mm_mul_ps(mpv, *tp++);
      ((dpc)[(q)*3 + p7X_I]) = _mm_fmadd_ps(ipv, *tp++, sv);
    }
    dcv = esl_sse_rightshift_ps(dcv, zerov);
    ((dpc)[(0) * 3 + p7X_D]) = zerov;
    tp = om->tfv + 7 * Q;
    for (q = 0; q < Q; q++)
    {
      ((dpc)[(q)*3 + p7X_D]) = _mm_add_ps(dcv, ((dpc)[(q)*3 + p7X_D]));
      dcv = _mm_mul_ps(((dpc)[(q)*3 + p7X_D]), *tp++);
    }
    if (om->M < 100)
    {
      for (j = 1; j < 4; j++)
      {
        dcv = esl_sse_rightshift_ps(dcv, zerov);
        tp = om->tfv + 7 * Q;
        for (q = 0; q < Q; q++)
        {
          ((dpc)[(q)*3 + p7X_D]) = _mm_add_ps(dcv, ((dpc)[(q)*3 + p7X_D]));
          dcv = _mm_mul_ps(dcv, *tp++);
        }
      }
    }
    else
    {
      for (j = 1; j < 4; j++)
      {
        register __m128 cv;

        dcv = esl_sse_rightshift_ps(dcv, zerov);
        tp = om->tfv + 7 * Q;
        cv = zerov;
        
        for (q = 0; q < Q; q++)
        {
          sv = _mm_add_ps(dcv, ((dpc)[(q)*3 + p7X_D]));
          cv = _mm_or_ps(cv, _mm_cmpgt_ps(sv, ((dpc)[(q)*3 + p7X_D])));
          ((dpc)[(q)*3 + p7X_D]) = sv;
          dcv = _mm_mul_ps(dcv, *tp++);
        }
        if (!_mm_movemask_ps(cv))
          break;
      }
    }

    for (q = 0; q < Q; q++)
      xEv = _mm_add_ps(((dpc)[(q)*3 + p7X_D]), xEv);

    xEv = _mm_add_ps(xEv,
                     ((__m128)__builtin_ia32_shufps((__v4sf)(__m128)(xEv), (__v4sf)(__m128)(xEv), (int)((((0) << 6) | ((3) << 4) | ((2) << 2) | (1))))));
    xEv = _mm_add_ps(xEv,
                     ((__m128)__builtin_ia32_shufps((__v4sf)(__m128)(xEv), (__v4sf)(__m128)(xEv), (int)((((1) << 6) | ((0) << 4) | ((3) << 2) | (2))))));
    _mm_store_ss(&xE, xEv);

    xN = xN * om->xf[p7O_N][p7O_LOOP];
    xC = (xC * om->xf[p7O_C][p7O_LOOP]) + (xE * om->xf[p7O_E][p7O_MOVE]);
    xJ = (xJ * om->xf[p7O_J][p7O_LOOP]) + (xE * om->xf[p7O_E][p7O_LOOP]);
    xB = (xJ * om->xf[p7O_J][p7O_MOVE]) + (xN * om->xf[p7O_N][p7O_MOVE]);

    if (xE > 1.0e4)
    {
      xN = xN / xE;
      xC = xC / xE;
      xJ = xJ / xE;
      xB = xB / xE;
      xEv = _mm_set1_ps(1.0 / xE);
      for (q = 0; q < Q; q++)
      {
        ((dpc)[(q)*3 + p7X_M]) = _mm_mul_ps(((dpc)[(q)*3 + p7X_M]), xEv);
        ((dpc)[(q)*3 + p7X_D]) = _mm_mul_ps(((dpc)[(q)*3 + p7X_D]), xEv);
        ((dpc)[(q)*3 + p7X_I]) = _mm_mul_ps(((dpc)[(q)*3 + p7X_I]), xEv);
      }
      ox->xmx[i * 6 + p7X_SCALE] = xE;
      ox->totscale += log(xE);
      xE = 1.0;
    }
    else
      ox->xmx[i * 6 + p7X_SCALE] = 1.0;

    ox->xmx[i * 6 + p7X_E] = xE;
    ox->xmx[i * 6 + p7X_N] = xN;
    ox->xmx[i * 6 + p7X_J] = xJ;
    ox->xmx[i * 6 + p7X_B] = xB;
    ox->xmx[i * 6 + p7X_C] = xC;
  }

  if (
      __builtin_isnan(
          xC))
    do
    {
      esl_exception(16, 0, "fwdback.c", 457, "forward score is NaN");
      return 16;
    } while (0);
  else if (L > 0 && xC == 0.0)
    do
    {
      esl_exception(16, 0, "fwdback.c", 458, "forward score underflow (is 0.0)");
      return 16;
    } while (0);
  else if (
      __builtin_isinf_sign(
          xC) == 1)
    do
    {
      esl_exception(16, 0, "fwdback.c", 459, "forward score overflow (is infinity)");
      return 16;
    } while (0);

  if (opt_sc !=
      ((void *)0))
    *opt_sc = ox->totscale + log(xC * om->xf[p7O_C][p7O_MOVE]);
  return 0;
}
int
forward_engine_old(int do_full, const ESL_DSQ *dsq, int L, const P7_OPROFILE *om, P7_OMX *ox, float *opt_sc)
{
  register __m128 mpv, dpv, ipv;
  register __m128 sv;
  register __m128 dcv;
  register __m128 xEv;
  register __m128 xBv;
  __m128 zerov;
  float xN, xE, xB, xC, xJ;
  int i;
  int q;
  int j;
  int Q = ((((2) > (((((om->M) - 1) / 4) + 1))) ? (2) : (((((om->M) - 1) / 4) + 1))));
  __m128 *dpc = ox->dpf[0];
  __m128 *dpp;
  __m128 *rp;
  __m128 *tp;

  ox->M = om->M;
  ox->L = L;
  ox->has_own_scales = 1;
  zerov = _mm_setzero_ps();
  for (q = 0; q < Q; q++)
    ((dpc)[(q)*3 + p7X_M]) = ((dpc)[(q)*3 + p7X_I]) = ((dpc)[(q)*3 + p7X_D]) = zerov;
  xE = ox->xmx[p7X_E] = 0.;
  xN = ox->xmx[p7X_N] = 1.;
  xJ = ox->xmx[p7X_J] = 0.;
  xB = ox->xmx[p7X_B] = om->xf[p7O_N][p7O_MOVE];
  xC = ox->xmx[p7X_C] = 0.;

  ox->xmx[p7X_SCALE] = 1.0;
  ox->totscale = 0.0;

  for (i = 1; i <= L; i++)
  {
    dpp = dpc;
    dpc = ox->dpf[do_full * i];
    rp = om->rfv[dsq[i]];
    tp = om->tfv;
    dcv = _mm_setzero_ps();
    xEv = _mm_setzero_ps();
    xBv = _mm_set1_ps(xB);

    mpv = esl_sse_rightshift_ps(((dpp)[(Q - 1) * 3 + p7X_M]), zerov);
    dpv = esl_sse_rightshift_ps(((dpp)[(Q - 1) * 3 + p7X_D]), zerov);
    ipv = esl_sse_rightshift_ps(((dpp)[(Q - 1) * 3 + p7X_I]), zerov);

    for (q = 0; q < Q; q++)
    {

      sv = _mm_mul_ps(xBv, *tp++);
      sv = _mm_add_ps(sv, _mm_mul_ps(mpv, *tp++));
      // sv = _mm_fmadd_ps(mpv, *tp++, sv);
      sv = _mm_add_ps(sv, _mm_mul_ps(ipv, *tp++));
      // sv = _mm_fmadd_ps(ipv, *tp++, sv);
      sv = _mm_add_ps(sv, _mm_mul_ps(dpv, *tp++));
      // sv = _mm_fmadd_ps(dpv, *tp++, sv);
      sv = _mm_mul_ps(sv, *rp++);
      xEv = _mm_add_ps(xEv, sv);

      mpv = ((dpp)[(q)*3 + p7X_M]);
      dpv = ((dpp)[(q)*3 + p7X_D]);
      ipv = ((dpp)[(q)*3 + p7X_I]);

      ((dpc)[(q)*3 + p7X_M]) = sv;
      ((dpc)[(q)*3 + p7X_D]) = dcv;

      dcv = _mm_mul_ps(sv, *tp++);

      sv = _mm_mul_ps(mpv, *tp++);
      //((dpc)[(q)*3 + p7X_I]) = _mm_add_ps(sv, _mm_mul_ps(ipv, *tp++));
      ((dpc)[(q)*3 + p7X_I]) = _mm_fmadd_ps(ipv, *tp++, sv);
    }
    dcv = esl_sse_rightshift_ps(dcv, zerov);
    ((dpc)[(0) * 3 + p7X_D]) = zerov;
    tp = om->tfv + 7 * Q;
    for (q = 0; q < Q; q++)
    {
      ((dpc)[(q)*3 + p7X_D]) = _mm_add_ps(dcv, ((dpc)[(q)*3 + p7X_D]));
      dcv = _mm_mul_ps(((dpc)[(q)*3 + p7X_D]), *tp++);
    }
    if (om->M < 100)
    {
      for (j = 1; j < 4; j++)
      {
        dcv = esl_sse_rightshift_ps(dcv, zerov);
        tp = om->tfv + 7 * Q;
        for (q = 0; q < Q; q++)
        {
          ((dpc)[(q)*3 + p7X_D]) = _mm_add_ps(dcv, ((dpc)[(q)*3 + p7X_D]));
          dcv = _mm_mul_ps(dcv, *tp++);
        }
      }
    }
    else
    {
      for (j = 1; j < 4; j++)
      {
        register __m128 cv;

        dcv = esl_sse_rightshift_ps(dcv, zerov);
        tp = om->tfv + 7 * Q;
        cv = zerov;
        for (q = 0; q < Q; q++)
        {
          sv = _mm_add_ps(dcv, ((dpc)[(q)*3 + p7X_D]));
          cv = _mm_or_ps(cv, _mm_cmpgt_ps(sv, ((dpc)[(q)*3 + p7X_D])));
          ((dpc)[(q)*3 + p7X_D]) = sv;
          dcv = _mm_mul_ps(dcv, *tp++);
        }
        if (!_mm_movemask_ps(cv))
          break;
      }
    }

    for (q = 0; q < Q; q++)
      xEv = _mm_add_ps(((dpc)[(q)*3 + p7X_D]), xEv);

    xEv = _mm_add_ps(xEv,
                     ((__m128)__builtin_ia32_shufps((__v4sf)(__m128)(xEv), (__v4sf)(__m128)(xEv), (int)((((0) << 6) | ((3) << 4) | ((2) << 2) | (1))))));
    xEv = _mm_add_ps(xEv,
                     ((__m128)__builtin_ia32_shufps((__v4sf)(__m128)(xEv), (__v4sf)(__m128)(xEv), (int)((((1) << 6) | ((0) << 4) | ((3) << 2) | (2))))));
    _mm_store_ss(&xE, xEv);

    xN = xN * om->xf[p7O_N][p7O_LOOP];
    xC = (xC * om->xf[p7O_C][p7O_LOOP]) + (xE * om->xf[p7O_E][p7O_MOVE]);
    xJ = (xJ * om->xf[p7O_J][p7O_LOOP]) + (xE * om->xf[p7O_E][p7O_LOOP]);
    xB = (xJ * om->xf[p7O_J][p7O_MOVE]) + (xN * om->xf[p7O_N][p7O_MOVE]);

    if (xE > 1.0e4)
    {
      xN = xN / xE;
      xC = xC / xE;
      xJ = xJ / xE;
      xB = xB / xE;
      xEv = _mm_set1_ps(1.0 / xE);
      for (q = 0; q < Q; q++)
      {
        ((dpc)[(q)*3 + p7X_M]) = _mm_mul_ps(((dpc)[(q)*3 + p7X_M]), xEv);
        ((dpc)[(q)*3 + p7X_D]) = _mm_mul_ps(((dpc)[(q)*3 + p7X_D]), xEv);
        ((dpc)[(q)*3 + p7X_I]) = _mm_mul_ps(((dpc)[(q)*3 + p7X_I]), xEv);
      }
      ox->xmx[i * 6 + p7X_SCALE] = xE;
      ox->totscale += log(xE);
      xE = 1.0;
    }
    else
      ox->xmx[i * 6 + p7X_SCALE] = 1.0;

    ox->xmx[i * 6 + p7X_E] = xE;
    ox->xmx[i * 6 + p7X_N] = xN;
    ox->xmx[i * 6 + p7X_J] = xJ;
    ox->xmx[i * 6 + p7X_B] = xB;
    ox->xmx[i * 6 + p7X_C] = xC;
  }

  if (
      __builtin_isnan(
          xC))
    do
    {
      esl_exception(16, 0, "fwdback.c", 457, "forward score is NaN");
      return 16;
    } while (0);
  else if (L > 0 && xC == 0.0)
    do
    {
      esl_exception(16, 0, "fwdback.c", 458, "forward score underflow (is 0.0)");
      return 16;
    } while (0);
  else if (
      __builtin_isinf_sign(
          xC) == 1)
    do
    {
      esl_exception(16, 0, "fwdback.c", 459, "forward score overflow (is infinity)");
      return 16;
    } while (0);

  if (opt_sc !=
      ((void *)0))
    *opt_sc = ox->totscale + log(xC * om->xf[p7O_C][p7O_MOVE]);
  return 0;
}
