/**
 * @file find_function.h
 * @author Jiaoyi
 * @brief find a record
 * @version 0.1
 * @date 2021-03-11
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef SRC_INCLUDE_FUNC_FIND_FUNCTION_H_
#define SRC_INCLUDE_FUNC_FIND_FUNCTION_H_

#include <float.h>

#include <algorithm>
#include <cstring>

#include "../carmi.h"
#include "./inlineFunction.h"

template <typename KeyType, typename ValueType>
BaseNode<KeyType> *CARMI<KeyType, ValueType>::Find(KeyType key, int *currunion,
                                                   int *currslot) {
  int idx = 0;
  int type = rootType;
  int fetch_start = 0;
  double fetch_leafIdx;
  while (1) {
    switch (type) {
      case PLR_ROOT_NODE:
        idx = root.childLeft +
              root.PLRType<DataVectorType, KeyType>::model.Predict(key);
        if (isPrimary == false) {
          fetch_leafIdx =
              root.PLRType<DataVectorType, KeyType>::model.PredictIdx(key);
          fetch_start = root.PLRType<DataVectorType, KeyType>::fetch_model
                            .PrefetchPredict(fetch_leafIdx);
          __builtin_prefetch(&entireData[fetch_start], 0, 3);
          // __builtin_prefetch(&entireData[fetch_start] + 64, 0, 3);
          // __builtin_prefetch(&entireData[fetch_start] + 128, 0, 3);
          // __builtin_prefetch(&entireData[fetch_start] + 192, 0, 3);
        }
        type = entireChild[idx].lr.flagNumber >> 24;
        break;
      case LR_INNER_NODE:
        idx = entireChild[idx].lr.childLeft + entireChild[idx].lr.Predict(key);
        type = entireChild[idx].lr.flagNumber >> 24;
        break;
      case PLR_INNER_NODE:
        idx =
            entireChild[idx].plr.childLeft + entireChild[idx].plr.Predict(key);
        type = entireChild[idx].lr.flagNumber >> 24;
        break;
      case HIS_INNER_NODE:
        idx =
            entireChild[idx].his.childLeft + entireChild[idx].his.Predict(key);
        type = entireChild[idx].lr.flagNumber >> 24;
        break;
      case BS_INNER_NODE:
        idx = entireChild[idx].bs.childLeft + entireChild[idx].bs.Predict(key);
        type = entireChild[idx].lr.flagNumber >> 24;
        break;
      case ARRAY_LEAF_NODE: {
        int left = entireChild[idx].array.m_left;

        *currunion = entireChild[idx].array.Predict(key);
        int find_idx = left + *currunion;

        // access entireData
        int res = SlotsUnionSearch(entireData[find_idx], key);

        if (entireData[find_idx].slots[res].first == key) {
          *currslot = res;
          return &entireChild[idx];
        } else {
          *currslot = -1;
          return &entireChild[idx];
        }
      }
      case EXTERNAL_ARRAY_LEAF_NODE: {
        auto size = entireChild[idx].externalArray.flagNumber & 0x00FFFFFF;
        int preIdx = entireChild[idx].externalArray.Predict(key);
        auto left = entireChild[idx].externalArray.m_left;

        if (*reinterpret_cast<const KeyType *>(
                static_cast<const char *>(external_data) +
                (left + preIdx) * recordLength) == key) {
          *currslot = preIdx;
          return &entireChild[idx];
        }

        preIdx = ExternalSearch(
            key, preIdx, entireChild[idx].externalArray.error, left, size);

        if (preIdx >= left + size ||
            *reinterpret_cast<const KeyType *>(
                static_cast<const char *>(external_data) +
                preIdx * recordLength) != key) {
          *currslot = 0;
          return NULL;
        }
        *currslot = preIdx - left;
        return &entireChild[idx];
      }
    }
  }
}

#endif  // SRC_INCLUDE_FUNC_FIND_FUNCTION_H_
