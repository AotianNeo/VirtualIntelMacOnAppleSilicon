// SPDX-License-Identifier: MIT
// Minimal adapter bridging Box64's arm64_emitter.h directly onto CEngine's E().
#ifndef __ARM64_MAPPING_H__
#define __ARM64_MAPPING_H__

#include <stdint.h>

#ifndef EMIT
#define EMIT(A) E((u32)(A))
#endif
#ifndef FEMIT
#define FEMIT(A) EMIT(A)
#endif
#ifndef LEMIT
#define LEMIT(A) EMIT(A)
#endif

#define xLR 30
#define xSP 31
#define xZR 31
#define wZR xZR

#define cEQ 0b0000
#define cNE 0b0001
#define cCS 0b0010
#define cHS cCS
#define cCC 0b0011
#define cLO cCC
#define cMI 0b0100
#define cPL 0b0101
#define cVS 0b0110
#define cVC 0b0111
#define cHI 0b1000
#define cLS 0b1001
#define cGE 0b1010
#define cLT 0b1011
#define cGT 0b1100
#define cLE 0b1101
#define c__ 0b1110

#endif // __ARM64_MAPPING_H__
