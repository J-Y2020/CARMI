/**
 * @file inlineFunction.h
 * @author Jiaoyi
 * @brief the inline functions for public functions
 * @version 0.1
 * @date 2021-03-11
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef SRC_INCLUDE_FUNC_INLINEFUNCTION_H_
#define SRC_INCLUDE_FUNC_INLINEFUNCTION_H_

#include <float.h>

#include <algorithm>
#include <vector>

#include "../carmi.h"
#include "../construct/minor_function.h"
#include "../params.h"

template <typename KeyType, typename ValueType>
inline int CARMI<KeyType, ValueType>::ArrayBinarySearch(double key, int start,
                                                        int end) const {
  while (start < end) {
    int mid = (start + end) / 2;
    if (entireData[mid].first < key)
      start = mid + 1;
    else
      end = mid;
  }
  return start;
}

template <typename KeyType, typename ValueType>
inline int CARMI<KeyType, ValueType>::ExternalBinarySearch(double key,
                                                           int start,
                                                           int end) const {
  while (start < end) {
    int mid = (start + end) / 2;
    if (*reinterpret_cast<const KeyType *>(
            static_cast<const char *>(external_data) + mid * recordLength) <
        key)
      start = mid + 1;
    else
      end = mid;
  }
  return start;
}

template <typename KeyType, typename ValueType>
inline int CARMI<KeyType, ValueType>::SlotsUnionSearch(
    const LeafSlots<KeyType, ValueType> &node, KeyType key) {
  int i = kMaxSlotNum / 2 - 1;
  int end = kMaxSlotNum;
  if (key <= node.slots[i].first || node.slots[i].first == -DBL_MAX) {
    end = i + 1;
    i = 0;
  } else {
    i++;
  }
  for (; i < end; i++) {
    if (key <= node.slots[i].first || node.slots[i].first == -DBL_MAX) {
      return i;
    }
  }
  return kMaxSlotNum - 1;
}

template <typename KeyType, typename ValueType>
inline bool CARMI<KeyType, ValueType>::ArrayInsertNext(
    const DataType &data, int nowDataIdx, int currunion,
    BaseNode<KeyType> *node) {
  if (data.first > entireData[nowDataIdx].slots[kMaxSlotNum - 1].first) {
    SlotsUnionInsert(data, currunion + 1, &entireData[nowDataIdx + 1], node);
    node->array.slotkeys[currunion] = entireData[nowDataIdx + 1].slots[0].first;

    return true;
  } else if (data.first < entireData[nowDataIdx].slots[kMaxSlotNum - 1].first) {
    SlotsUnionInsert(entireData[nowDataIdx].slots[kMaxSlotNum - 1],
                     currunion + 1, &entireData[nowDataIdx + 1], node);

    if (entireData[nowDataIdx].slots[0].first > data.first) {
      for (int j = kMaxSlotNum - 1; j > 0; j--) {
        entireData[nowDataIdx].slots[j] = entireData[nowDataIdx].slots[j - 1];
      }
      entireData[nowDataIdx].slots[0] = data;
      if (currunion != 0)
        node->array.slotkeys[currunion - 1] =
            entireData[nowDataIdx].slots[0].first;
    } else {
      for (int i = kMaxSlotNum - 2; i >= 0; i--) {
        if (entireData[nowDataIdx].slots[i].first <= data.first) {
          for (int j = kMaxSlotNum - 1; j > i + 1; j--) {
            entireData[nowDataIdx].slots[j] =
                entireData[nowDataIdx].slots[j - 1];
          }
          entireData[nowDataIdx].slots[i + 1] = data;
          break;
        }
      }
    }
    node->array.slotkeys[currunion] = entireData[nowDataIdx + 1].slots[0].first;
    return true;
  }
  return false;
}

template <typename KeyType, typename ValueType>
inline int CARMI<KeyType, ValueType>::ArraySearch(double key, int preIdx,
                                                  int error, int left,
                                                  int size) const {
  int start = std::max(0, preIdx - error) + left;
  int end = std::min(std::max(0, size - 1), preIdx + error) + left;
  start = std::min(start, end);
  int res;
  if (key <= entireData[start].first)
    res = ArrayBinarySearch(key, left, start);
  else if (key <= entireData[end].first)
    res = ArrayBinarySearch(key, start, end);
  else
    res = ArrayBinarySearch(key, end, left + size - 1);
  return res;
}

template <typename KeyType, typename ValueType>
inline int CARMI<KeyType, ValueType>::ExternalSearch(double key, int preIdx,
                                                     int error, int left,
                                                     int size) const {
  int start = std::max(0, preIdx - error) + left;
  int end = std::min(std::max(0, size - 1), preIdx + error) + left;
  start = std::min(start, end);
  int res;
  if (key <=
      *reinterpret_cast<const KeyType *>(
          static_cast<const char *>(external_data) + start * recordLength))
    res = ExternalBinarySearch(key, left, start);
  else if (key <=
           *reinterpret_cast<const KeyType *>(
               static_cast<const char *>(external_data) + end * recordLength))
    res = ExternalBinarySearch(key, start, end);
  else
    res = ExternalBinarySearch(key, end, left + size - 1);
  return res;
}

template <typename KeyType, typename ValueType>
template <typename TYPE>
inline void CARMI<KeyType, ValueType>::Split(bool isExternal, int left,
                                             int size, int previousIdx,
                                             int idx) {
  int actualSize = 0;
  DataVectorType tmpDataset = ExtractData(left, size + left, &actualSize);

  int nextIdx = entireChild[idx].array.nextLeaf;

  // create a new inner node
  auto node = LRModel();
  int childNum = kInsertNewChildNumber;
  node.SetChildNumber(childNum);
  node.childLeft = AllocateChildMemory(childNum);
  Train(0, actualSize, tmpDataset, &node);
  entireChild[idx].lr = node;

  std::vector<IndexPair> perSize(childNum, emptyRange);
  IndexPair range(0, actualSize);
  NodePartition<LRModel>(node, range, tmpDataset, &perSize);

  int tmpLeft = left;
  for (int i = 0; i < childNum; i++) {
    TYPE tmpLeaf;
    Init(perSize[i].left, perSize[i].size, tmpDataset, &tmpLeaf);

    if (isExternal) {
      tmpLeaf.m_left = tmpLeft;
      tmpLeft += perSize[i].size;
    }
    entireChild[node.childLeft + i].array =
        *(reinterpret_cast<ArrayType<KeyType> *>(&tmpLeaf));
  }

  if (!isExternal) {
    if (previousIdx >= 0) {
      entireChild[previousIdx].array.nextLeaf = node.childLeft;
    }
    entireChild[node.childLeft].array.previousLeaf = previousIdx;
    int end = node.childLeft + childNum - 1;
    for (int i = node.childLeft + 1; i < end; i++) {
      entireChild[i].array.previousLeaf = i - 1;
      entireChild[i].array.nextLeaf = i + 1;
    }
    entireChild[end].array.previousLeaf = end - 1;
    if (nextIdx != -1) {
      entireChild[end].array.nextLeaf = nextIdx;
      entireChild[nextIdx].array.previousLeaf = end;
    }
  }
}

#endif  // SRC_INCLUDE_FUNC_INLINEFUNCTION_H_
