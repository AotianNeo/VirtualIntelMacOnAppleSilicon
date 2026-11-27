#pragma once
#include <cstdio>
#include <cstdlib>

#define LOGMAN_THROW_A_FMT(pred, ...) \
  do { \
    if (!(pred)) { \
      std::fprintf(stderr, "FEX CodeEmitter assert failed: %s\n", #pred); \
      std::abort(); \
    } \
  } while (0)

#define LOGMAN_MSG_A_FMT(...) \
  do { \
    std::fprintf(stderr, "FEX CodeEmitter abort\n"); \
    std::abort(); \
  } while (0)

#define ERROR_AND_DIE_FMT(...) LOGMAN_MSG_A_FMT(__VA_ARGS__)
