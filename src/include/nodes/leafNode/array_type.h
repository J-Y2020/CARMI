/**
 * @file array_type.h
 * @author Jiaoyi
 * @brief
 * @version 0.1
 * @date 2021-03-11
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef SRC_INCLUDE_NODES_LEAFNODE_ARRAY_TYPE_H_
#define SRC_INCLUDE_NODES_LEAFNODE_ARRAY_TYPE_H_

#include <float.h>

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "../../carmi.h"
#include "../../construct/minor_function.h"
#include "../../params.h"
#include "./leaf_nodes.h"

template <typename KeyType>
inline int ArrayType<KeyType>::Predict(KeyType key) const {
  // return the idx of the union in entireData
  int end_idx = (flagNumber & 0x00FFFFFF) - 2;
  if (end_idx < 0) {
    return 0;
  }
  for (int i = 0; i <= end_idx; i++) {
    if (key < slotkeys[i]) {
      return i;
    }
  }
  return end_idx + 1;
}

template <typename KeyType, typename ValueType>
inline void CARMI<KeyType, ValueType>::Init(int left, int size,
                                            const DataVectorType &dataset,
                                            ArrayType<KeyType> *arr) {
  int actualSize = 0;
  DataVectorType newDataset = ExtractData(left, size, dataset, &actualSize);
  int neededLeafNum = GetActualSize(size);

  StoreData(neededLeafNum, 0, actualSize, newDataset, arr);
}

template <typename KeyType, typename ValueType>
inline void CARMI<KeyType, ValueType>::Rebalance(const int unionleft,
                                                 const int unionright,
                                                 ArrayType<KeyType> *arr) {
  int actualSize = 0;
  DataVectorType newDataset = ExtractData(unionleft, unionright, &actualSize);
  int nowLeafNum = unionright - unionleft;

  StoreData(nowLeafNum, 0, actualSize, newDataset, arr);
}

template <typename KeyType, typename ValueType>
inline void CARMI<KeyType, ValueType>::Expand(const int unionleft,
                                              const int unionright,
                                              ArrayType<KeyType> *arr) {
  int actualSize = 0;
  DataVectorType newDataset = ExtractData(unionleft, unionright, &actualSize);
  int neededLeafNum = unionright - unionleft + 1;

  StoreData(neededLeafNum, 0, actualSize, newDataset, arr);
}

template <typename KeyType, typename ValueType>
inline bool CARMI<KeyType, ValueType>::CheckIsPrefetch(
    int neededLeafNum, int left, int size, const DataVectorType &dataset) {
  int end = left + size;
  double predictLeafIdx = root.model.PredictIdx(dataset[left].first);
  int leftIdx = root.fetch_model.PrefetchPredict(predictLeafIdx);
  // check whether these points can be prefetched
  if (leftIdx + neededLeafNum > entireData.size()) {
    return false;
  }
  predictLeafIdx = root.model.PredictIdx(dataset[end - 1].first);
  int rightIdx = root.fetch_model.PrefetchPredict(predictLeafIdx);
  rightIdx = std::min(leftIdx + neededLeafNum - 1, rightIdx);
  for (int i = leftIdx; i <= rightIdx; i++) {
    if (entireData[i].slots[0].first != -DBL_MAX) {
      return false;
    }
  }
  return true;
}

template <typename KeyType, typename ValueType>
inline bool CARMI<KeyType, ValueType>::StorePrevious(
    const DataVectorType &dataset, int neededBlockNum,
    std::vector<LeafSlots<KeyType, ValueType>> *tmpBlockVec,
    int *actualBlockNum, int *missNumber) {
  int maxInsertedNumber = readRate * kMaxSlotNum;
  int size = dataset.size();
  if (size > neededBlockNum * maxInsertedNumber) {
    return false;
  }
  std::map<KeyType, int> actualIdx;
  *actualBlockNum = 0;
  int leftIdx =
      root.fetch_model.PrefetchPredict(root.model.PredictIdx(dataset[0].first));
  BaseNode<KeyType> tmpArr;
  for (int i = 0; i < size; i++) {
    double predictLeafIdx = root.model.PredictIdx(dataset[i].first);
    int p = root.fetch_model.PrefetchPredict(predictLeafIdx);
    p -= leftIdx;
    if (p >= neededBlockNum) {
      p = neededBlockNum - 1;
    }
    auto insertSuccess =
        SlotsUnionInsert(dataset[i], 0, &(*tmpBlockVec)[p], &tmpArr);
    DataType nowDatapoint = dataset[i];
    DataType preDatapoint;
    std::vector<LeafSlots<KeyType, ValueType>> tmp = *tmpBlockVec;
    int tmpP = p;

    // insert into the previous block
    while (!insertSuccess) {
      preDatapoint = (*tmpBlockVec)[p].slots[0];
      for (int j = 0; j < maxInsertedNumber - 1; j++) {
        (*tmpBlockVec)[p].slots[j] = (*tmpBlockVec)[p].slots[j + 1];
      }
      (*tmpBlockVec)[p].slots[maxInsertedNumber - 1] = nowDatapoint;
      p--;
      if (p < 0) {
        break;
      }
      insertSuccess =
          SlotsUnionInsert(preDatapoint, 0, &(*tmpBlockVec)[p], &tmpArr);
      nowDatapoint = preDatapoint;
    }
    // insert into the subsequent block
    if (!insertSuccess) {
      p = tmpP;
      while (p + 1 < neededBlockNum && !insertSuccess) {
        *tmpBlockVec = tmp;
        p++;
        insertSuccess =
            SlotsUnionInsert(dataset[i], 0, &(*tmpBlockVec)[p], &tmpArr);
      }
    }
    if (insertSuccess) {
      if (p + 1 > *actualBlockNum) {
        *actualBlockNum = p + 1;
      }
    } else {
      return false;
    }
  }

  int tmpMissNum = 0;
  for (int i = 0; i < *actualBlockNum; i++) {
    for (int j = 0; j < maxInsertedNumber; j++) {
      if ((*tmpBlockVec)[i].slots[j].first != -DBL_MAX) {
        double preLeafIdx =
            root.model.PredictIdx((*tmpBlockVec)[i].slots[j].first);
        int fetchIdx = root.fetch_model.PrefetchPredict(preLeafIdx);
        fetchIdx -= leftIdx;
        if (fetchIdx != i) {
          tmpMissNum++;
        }
      }
    }
  }
  *missNumber = tmpMissNum;

  return true;
}

template <typename KeyType, typename ValueType>
inline bool CARMI<KeyType, ValueType>::StoreSubsequent(
    const DataVectorType &dataset, int neededBlockNum,
    std::vector<LeafSlots<KeyType, ValueType>> *tmpBlockVec,
    int *actualBlockNum, int *missNumber) {
  int size = dataset.size();
  int maxInsertedNumber = readRate * kMaxSlotNum;
  if (size > neededBlockNum * maxInsertedNumber) {
    return false;
  }
  std::map<KeyType, int> actualIdx;
  *actualBlockNum = 0;
  int leftIdx =
      root.fetch_model.PrefetchPredict(root.model.PredictIdx(dataset[0].first));
  BaseNode<KeyType> tmpArr;
  for (int i = 0; i < size; i++) {
    double predictLeafIdx = root.model.PredictIdx(dataset[i].first);
    int p = root.fetch_model.PrefetchPredict(predictLeafIdx);
    p -= leftIdx;
    if (p >= neededBlockNum) {
      p = neededBlockNum - 1;
    }
    auto insertSuccess =
        SlotsUnionInsert(dataset[i], 0, &(*tmpBlockVec)[p], &tmpArr);
    DataType nowDatapoint = dataset[i];
    DataType preDatapoint;
    // insert into the subsequent block
    while (!insertSuccess && p + 1 < neededBlockNum) {
      p++;
      insertSuccess =
          SlotsUnionInsert(dataset[i], 0, &(*tmpBlockVec)[p], &tmpArr);
    }

    // insert into the previous block
    while (!insertSuccess) {
      preDatapoint = (*tmpBlockVec)[p].slots[0];
      for (int j = 0; j < maxInsertedNumber - 1; j++) {
        (*tmpBlockVec)[p].slots[j] = (*tmpBlockVec)[p].slots[j + 1];
      }
      (*tmpBlockVec)[p].slots[maxInsertedNumber - 1] = nowDatapoint;
      p--;
      if (p < 0) {
        break;
      }
      insertSuccess =
          SlotsUnionInsert(preDatapoint, 0, &(*tmpBlockVec)[p], &tmpArr);
      nowDatapoint = preDatapoint;
    }

    if (insertSuccess) {
      if (p + 1 > *actualBlockNum) {
        *actualBlockNum = p + 1;
      }
    } else {
      return false;
    }
  }

  int tmpMissNum = 0;
  for (int i = 0; i < *actualBlockNum; i++) {
    for (int j = 0; j < maxInsertedNumber; j++) {
      if ((*tmpBlockVec)[i].slots[j].first != -DBL_MAX) {
        double preLeafIdx =
            root.model.PredictIdx((*tmpBlockVec)[i].slots[j].first);
        int fetchIdx = root.fetch_model.PrefetchPredict(preLeafIdx);
        fetchIdx -= leftIdx;
        if (fetchIdx != i) {
          tmpMissNum++;
        }
      }
    }
  }

  *missNumber = tmpMissNum;
  return true;
}

template <typename KeyType, typename ValueType>
inline double CARMI<KeyType, ValueType>::CalculateArrayCost(int size,
                                                            int totalSize,
                                                            int givenBlockNum) {
  double space = kBaseNodeSpace;
  double time = carmi_params::kLeafBaseTime;
  int maxInsertedNumber = readRate * kMaxSlotNum;
  if (givenBlockNum * maxInsertedNumber >= size) {
    // can be prefetched
    space += givenBlockNum * carmi_params::kMaxLeafNodeSize / 1024.0 / 1024.0;
  } else {
    // cannot be prefetched
    int neededBlock = GetActualSize(size);
    space += neededBlock * carmi_params::kMaxLeafNodeSize / 1024.0 / 1024.0;
    time += carmi_params::kMemoryAccessTime;
  }
  time *= size * 1.0 / totalSize;
  double cost = time + lambda * space;
  return cost;
}

template <typename KeyType, typename ValueType>
inline bool CARMI<KeyType, ValueType>::StoreData(int neededLeafNum, int left,
                                                 int size,
                                                 const DataVectorType &dataset,
                                                 ArrayType<KeyType> *arr) {
  if (neededLeafNum == 0 || size == 0) {
    arr->flagNumber = (ARRAY_LEAF_NODE << 24) + 0;
    return true;
  }
  int end = left + size;
  int actualNum = 0;
  bool isPossible = true;
  double predictLeafIdx = root.model.PredictIdx(dataset[left].first);
  int leftIdx = root.fetch_model.PrefetchPredict(predictLeafIdx);
  // check whether these points can be prefetched
  if (isInitMode) {
    isPossible = CheckIsPrefetch(neededLeafNum, left, size, dataset);
  }
  if (isInitMode && isPossible) {
    arr->m_left = leftIdx;

    DataVectorType tmpDataset(dataset.begin() + left, dataset.begin() + end);
    LeafSlots<KeyType, ValueType> tmpSlot;
    std::vector<LeafSlots<KeyType, ValueType>> prevBlocks(neededLeafNum,
                                                          tmpSlot);
    int prevActualNum = 0;
    int prevMissNum = 0;
    bool isPreviousSuccess = StorePrevious(
        tmpDataset, neededLeafNum, &prevBlocks, &prevActualNum, &prevMissNum);

    std::vector<LeafSlots<KeyType, ValueType>> nextBlocks(neededLeafNum,
                                                          tmpSlot);
    int nextActualNum = 0;
    int nextMissNum = 0;
    bool isNextSuccess = StoreSubsequent(tmpDataset, neededLeafNum, &nextBlocks,
                                         &nextActualNum, &nextMissNum);

    bool isPrev = true;
    if (isNextSuccess && isPreviousSuccess) {
      if (nextMissNum < prevMissNum) {
        isPrev = false;
      }
    } else if (isNextSuccess) {
      isPrev = false;
    } else if (isPreviousSuccess) {
      isPrev = true;
    } else {
      return false;
    }
    int tmpMissNum = 0;
    if (isPrev) {
      actualNum = prevActualNum;
      tmpMissNum = prevMissNum;
      for (int i = arr->m_left; i < arr->m_left + actualNum; i++) {
        entireData[i] = prevBlocks[i - arr->m_left];
      }
    } else {
      actualNum = nextActualNum;
      tmpMissNum = nextMissNum;
      for (int i = arr->m_left; i < arr->m_left + actualNum; i++) {
        entireData[i] = nextBlocks[i - arr->m_left];
      }
    }

    if (arr->m_left + actualNum > static_cast<int>(nowDataSize)) {
      nowDataSize = arr->m_left + actualNum;
    }
    if (arr->m_left - prefetchEnd > 1) {
      if (prefetchEnd < 0) {
        AllocateEmptyBlock(0, arr->m_left);
      } else {
        AllocateEmptyBlock(prefetchEnd + 1, arr->m_left - prefetchEnd - 1);
      }
    }
    prefetchEnd = arr->m_left + actualNum - 1;

  } else {
    // for EXPAND
    int nowLeafNum = arr->flagNumber & 0x00FFFFFF;
    if (nowLeafNum == 0) {
      arr->m_left = AllocateMemory(neededLeafNum);
    } else {
      if (nowLeafNum != neededLeafNum) {
        if (arr->m_left != -1) {
          ReleaseMemory(arr->m_left, nowLeafNum);
        }
        arr->m_left = AllocateMemory(neededLeafNum);
      }
    }

    LeafSlots<KeyType, ValueType> tmp;
    int avg = std::max(1.0, ceil(size * 1.0 / neededLeafNum));
    avg = std::min(avg, kMaxSlotNum);
    BaseNode<KeyType> tmpArr;
    entireData[arr->m_left] = LeafSlots<KeyType, ValueType>();

    for (int i = arr->m_left, j = left, k = 1; j < end; j++, k++) {
      SlotsUnionInsert(dataset[j], 0, &tmp, &tmpArr);
      if (k == avg || j == end - 1) {
        k = 0;
        entireData[i++] = tmp;
        tmp = LeafSlots<KeyType, ValueType>();
        actualNum++;
      }
    }
  }

  arr->flagNumber = (ARRAY_LEAF_NODE << 24) + actualNum;

  if (actualNum <= 1) {
    arr->slotkeys[0] = dataset[left + size - 1].first + 1;
    return true;
  }
  end = arr->m_left + actualNum;
  int j = 0;
  double lastKey = dataset[left].first;
  for (int i = 0; i < kMaxSlotNum; i++) {
    if (entireData[arr->m_left].slots[i].first != -DBL_MAX) {
      lastKey = entireData[arr->m_left].slots[i].first;
    } else {
      break;
    }
  }
  for (int i = arr->m_left + 1; i < end; i++, j++) {
    if (entireData[i].slots[0].first != -DBL_MAX) {
      arr->slotkeys[j] = entireData[i].slots[0].first;
      for (int k = kMaxSlotNum - 1; k >= 0; k--) {
        if (entireData[i].slots[k].first != -DBL_MAX) {
          lastKey = entireData[i].slots[k].first;
          break;
        }
      }
    } else {
      arr->slotkeys[j] = lastKey + 1;
    }
  }
  return true;
}

#endif  // SRC_INCLUDE_NODES_LEAFNODE_ARRAY_TYPE_H_
