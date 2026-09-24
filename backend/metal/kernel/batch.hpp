#pragma once

enum : unsigned long {
  convStateBytes = 8192 * 4 * 2,
  recurrentStateBytes = 32 * 128 * 128 * 4,
  gdnCheckpointBytes = 24 * (convStateBytes + recurrentStateBytes),
  stateBytes = gdnCheckpointBytes + 8192
};

// Target records stay immutable; MTP sampling advances positions in the separate drafter records.
struct GpuQuery {
  unsigned long previous, next;
  unsigned int start, count, position, slot, state, logit, samples;
  float temperature, topP;
  int topK;
};

#ifdef INFENG_METAL
inline device const GpuQuery& queryRow(device const GpuQuery* queries, unsigned int row) {
  while (row >= queries->start + queries->count)
    ++queries;
  return *queries;
}

inline device const GpuQuery& queryLogit(device const GpuQuery* queries, unsigned int row) {
  while (!queries->samples || row >= queries->logit + queries->samples)
    ++queries;
  return *queries;
}
#endif
