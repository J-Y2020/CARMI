/**
 * @file lr_model.h
 * @author Jiaoyi
 * @brief
 * @version 0.1
 * @date 2021-03-11
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef SRC_INCLUDE_NODES_INNERNODE_LR_MODEL_H_
#define SRC_INCLUDE_NODES_INNERNODE_LR_MODEL_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "../../carmi.h"
#include "../../construct/minor_function.h"

template <typename KeyType, typename ValueType>
inline void CARMI<KeyType, ValueType>::Train(const int left, const int size,
                                             const DataVectorType &dataset,
                                             LRModel *lr) {
  if (size == 0) return;

  // calculate divisor
  int childNumber = lr->flagNumber & 0x00FFFFFF;
  int end = left + size;
  double maxValue = dataset[end - 1].first;
  lr->minValue = dataset[left].first;
  lr->divisor = 1.0 / (static_cast<float>(maxValue - lr->minValue) / 6);

  // extract data points and their index
  std::vector<int> segCnt(6, 0);
  DataVectorType data(size, {-DBL_MAX, -DBL_MAX});
  for (int i = left, j = 0; i < end; i++, j++) {
    data[j].first = dataset[i].first;
    data[j].second = static_cast<double>(j) / size * childNumber;
    int idx = (dataset[i].first - lr->minValue) * lr->divisor;
    idx = std::min(idx, 5);
    idx = std::max(0, idx);
    segCnt[idx]++;
  }

  // train in segments
  int start_idx = left;
  for (int k = 0; k < 6; k++) {
    float a, b;
    LRTrain(start_idx, segCnt[k], dataset, &a, &b);
    int bound = childNumber / 6 * (k + 1);
    lr->theta[k] = {a * bound, b * bound};
    start_idx += segCnt[k];
  }
}

/**
 * @brief predict the index of the next branch
 *
 * @param key
 * @return int index (from 0 to childNumber-1 )
 */
inline int LRModel::Predict(double key) const {
  int idx = (key - minValue) * divisor;
  if (idx < 0)
    idx = 0;
  else if (idx >= 6)
    idx = 5;

  // return the predicted idx in the children
  int p = theta[idx].first * key + theta[idx].second;
  int bound = (flagNumber & 0x00FFFFFF) * 0.16666;
  idx = bound * idx;
  if (p < idx)
    p = idx;
  else if (p >= idx + bound)
    p = idx + bound - 1;
  return p;
}
#endif  // SRC_INCLUDE_NODES_INNERNODE_LR_MODEL_H_
