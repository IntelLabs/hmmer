
#include "easel.h"
#include "hmmer.h"
#include "p7_config.h"
#include <immintrin.h>
//#include <tbb/tbb.h>
//#include <cstring>
#include "calc_band.h"

//#include <bitset>
//#include <iostream>
//#include <algorithm>
//#include <atomic>
//#include <memory>

//void print_m512i(__m512i v)
//{
//  uint64_t buffer[8];
//  _mm512_storeu_si512((__m512 *)buffer, v);
//  for (int i = 0; i < 8; i++)
//  {
//    std::bitset<64> b(buffer[i]);
//    std::cout << i << ": " << b << std::endl;
//  }
//  std::cout << std::endl;
//}
//
//void print_m128i(__m128i v)
//{
//  uint64_t buffer[2];
//  _mm_storeu_si128((__m128i *)buffer, v);
//  for (int i = 0; i < 2; i++)
//  {
//    std::bitset<64> b(buffer[i]);
//    std::cout << i << ": " << b << std::endl;
//  }
//  std::cout << std::endl;
//}
//
//__m512i _mm512_set_m256i(__m256i hi, __m256i lo)
//{
//  __m512i base = _mm512_castsi256_si512(lo); // upper half is don't-care
//  return _mm512_inserti32x8(base, hi, 1);    // insert hi as new upper half
//}
//
//void print_128_num(__m128i var)
//{
//  uint16_t val[8];
//  memcpy(val, &var, sizeof(val));
//  printf("%i %i %i %i %i %i %i %i \n",
//         val[0], val[1], val[2], val[3], val[4], val[5],
//         val[6], val[7]);
//}
//
//
//void print_512_num(__m512i var)
//{
//  __m128i v1, v2;
//  __m256i xEv4_H = _mm512_extracti64x4_epi64(var, 1);
//  v1 = _mm256_extracti64x2_epi64(xEv4_H, 0);
//  v2 = _mm256_extracti64x2_epi64(xEv4_H, 1);
//  print_128_num(v2);
//  print_128_num(v1);
//  __m256i xEv4_L = _mm512_extracti64x4_epi64(var, 0);
//  v1 = _mm256_extracti64x2_epi64(xEv4_L, 0);
//  v2 = _mm256_extracti64x2_epi64(xEv4_L, 1);
//  print_128_num(v2);
//  print_128_num(v1);
//}
//

__m128i calc_band_avx512(const ESL_DSQ *dsq, const int L,
                                  const P7_OPROFILE *om, const int q, __m128i beginv,
                                  __m128i xEv, const int band)
{
  int i;
  int i2;
  const int Q = p7O_NQB(om->M);
  register __m512i *rsc;

  dsq++;
  const __m512i beginvx4 = _mm512_broadcast_i32x4(beginv);
  const uint64_t m64 = 0xFFFFFFFFFFFFFFFF;
  /*
  register const __m512i masks[] = {
      _mm512_set_epi64(m64, m64, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0),
      _mm512_set_epi64(0x0, 0x0, m64, m64, 0x0, 0x0, 0x0, 0x0),
      _mm512_set_epi64(0x0, 0x0, 0x0, 0x0, m64, m64, 0x0, 0x0),
      _mm512_set_epi64(0x0, 0x0, 0x0, 0x0, 0x0, 0x0, m64, m64)
      };
  register const __m512i inverse_masks[] = {
      _mm512_set_epi64(0x0, 0x0, m64, m64, m64, m64, m64, m64),
      _mm512_set_epi64(m64, m64, 0x0, 0x0, m64, m64, m64, m64),
      _mm512_set_epi64(m64, m64, m64, m64, 0x0, 0x0, m64, m64),
      _mm512_set_epi64(m64, m64, m64, m64, m64, m64, 0x0, 0x0)};
  */
  __mmask8 mask8[] = {0xC0, 0x30, 0xC, 0x3}; 

  __m512i sv[] = {beginvx4, beginvx4, beginvx4};

  __m512i xEv4;

  __m512i data;

  for (i = 0; i < L && i < Q - q - band; i++)
  {
    rsc = (__m512i *)(om->sbv[dsq[i]] + i + q);
    for (int j = 0; j < 3; j++)
    {
      sv[j] = _mm512_subs_epi8(sv[j], *rsc++);
      xEv4 = _mm512_max_epu8(xEv4, sv[j]);
    }
  }
  
  i = Q - q - band;
  for (int j = band; j >= 1 && i < L; j--, i++)
  {
    rsc = (__m512i *)(om->sbv[dsq[i]] + Q - j);
    for (int k = 0; k < 3; k++)
    {
      sv[k] = _mm512_subs_epi8(sv[k], *rsc++);
      xEv4 = _mm512_max_epu8(xEv4, sv[k]);
    }
    int sv_index = (j+3)/4-1;
    int mask_index = (band-j) %4;
    data =_mm512_bslli_epi128(sv[sv_index], 1);
    data = _mm512_or_epi64(data, beginvx4);
    /*    
    data = _mm512_and_epi64(data, masks[mask_index]);
    sv[sv_index] = _mm512_and_epi64(sv[sv_index], inverse_masks[mask_index]);
    sv[sv_index] = _mm512_or_epi64(sv[sv_index], data);
    */
    sv[sv_index] = _mm512_mask_mov_epi64(sv[sv_index], mask8[mask_index], data);
  }


  for (i2 = Q - q; i2 < L - Q; i2 += Q)
  {
    for (i = 0; i < Q - band; i++)
    {
      rsc = (__m512i *)(om->sbv[dsq[i2 + i]] + i);
      for (int k = 0; k < 3; k++)
      {
        sv[k] = _mm512_subs_epi8(sv[k], *rsc++);
        xEv4 = _mm512_max_epu8(xEv4, sv[k]);
      }
    }
    i += i2;

    for (int j = band; j >= 1 && i < L; j--, i++)
    {
      rsc = (__m512i *)(om->sbv[dsq[i]] + Q - j);
      for (int k = 0; k < 3; k++)
      {
        sv[k] = _mm512_subs_epi8(sv[k], *rsc++);
        xEv4 = _mm512_max_epu8(xEv4, sv[k]);
      }

      int sv_index = (j+3)/4-1;
      int mask_index = (band-j) %4;
      data =_mm512_bslli_epi128(sv[sv_index], 1);
      data = _mm512_or_epi64(data, beginvx4);
      /*
      data = _mm512_and_epi64(data, masks[mask_index]);
      sv[sv_index] = _mm512_and_epi64(sv[sv_index], inverse_masks[mask_index]);
      sv[sv_index] = _mm512_or_epi64(sv[sv_index], data);*/
      sv[sv_index] = _mm512_mask_mov_epi64(sv[sv_index], mask8[mask_index], data);
    }
  }

  for (i = 0; i2 + i < L && i < Q - band; i++)
  {
    rsc = (__m512i *)(om->sbv[dsq[i2 + i]] + i);
    for (int k = 0; k < 3; k++)
    {
      sv[k] = _mm512_subs_epi8(sv[k], *rsc++);
      xEv4 = _mm512_max_epu8(xEv4, sv[k]);
    }
  };
  i = i + i2;
  for (int j = band; j >= 1 && i < L; j--, i++)
  {
    rsc = (__m512i *)(om->sbv[dsq[i]] + Q - j);
    for (int k = 0; k < 3; k++)
    {
      sv[k] = _mm512_subs_epi8(sv[k], *rsc++);
      xEv4 = _mm512_max_epu8(xEv4, sv[k]);
    }

    int sv_index = (j+3)/4-1;
    int mask_index = (band-j) %4;
    data =_mm512_bslli_epi128(sv[sv_index], 1);
    data = _mm512_or_epi64(data, beginvx4);
    /*
    data = _mm512_and_epi64(data, masks[mask_index]);
    sv[sv_index] = _mm512_and_epi64(sv[sv_index], inverse_masks[mask_index]);
    sv[sv_index] = _mm512_or_epi64(sv[sv_index], data);
    */
   sv[sv_index] = _mm512_mask_mov_epi64(sv[sv_index], mask8[mask_index], data);
  }

  __m256i xEv4_L = _mm512_extracti64x4_epi64(xEv4, 0);
  __m256i xEv4_H = _mm512_extracti64x4_epi64(xEv4, 1);
  xEv4_L = _mm256_max_epu8(xEv4_L, xEv4_H);
  xEv = _mm256_extracti64x2_epi64(xEv4_L, 0);
  __m128i xEv2_H = _mm256_extracti64x2_epi64(xEv4_L, 1);
  xEv = _mm_max_epu8(xEv, xEv2_H);

  return xEv;
}

