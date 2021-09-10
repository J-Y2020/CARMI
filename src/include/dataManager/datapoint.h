/**
 * @file datapoint.h
 * @author Jiaoyi
 * @brief manage the entireData array
 * @version 0.1
 * @date 2021-03-11
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef SRC_INCLUDE_DATAMANAGER_DATAPOINT_H_
#define SRC_INCLUDE_DATAMANAGER_DATAPOINT_H_
#include <float.h>
#include <math.h>

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#include "../carmi.h"
#include "./empty_block.h"

template <typename KeyType, typename ValueType>
inline bool CARMI<KeyType, ValueType>::AllocateEmptyBlock(int left, int len) {
  if (len == 0) return true;
  int res = 0;
  for (int i = emptyBlocks.size() - 1; i > 0; i--) {
    res = emptyBlocks[i].addBlock(left, len);
    while (res > 0) {
      len = res;
      res = emptyBlocks[i].addBlock(left, len);
      if (res == -1) {
        break;
      }
    }
    if (res == 0) {
      return true;
    }
  }
  return false;
}

template <typename KeyType, typename ValueType>
inline int CARMI<KeyType, ValueType>::GetActualSize(int size) {
  if (size <= 0) return 0;
  int maxStoredNum = readRate * kMaxSlotNum;
  if (size > kMaxLeafNum * maxStoredNum) {
#ifdef DEBUG
    std::cout << "the size is " << size
              << ",\tthe maximum size in a leaf node is "
              << kMaxLeafNum * maxStoredNum << std::endl;
#endif  // DEBUG
    return -1;
  }
  int leafNumber =
      std::min(static_cast<int>(ceil(size * 1.0 / maxStoredNum)), kMaxLeafNum);

  return leafNumber;
}

template <typename KeyType, typename ValueType>
void CARMI<KeyType, ValueType>::InitEntireData(int size) {
  entireDataSize = std::max(64.0, size / carmi_params::kMaxLeafNodeSize * 1.5);
  std::vector<LeafSlots<KeyType, ValueType>>().swap(entireData);
  entireData = std::vector<LeafSlots<KeyType, ValueType>>(
      entireDataSize, LeafSlots<KeyType, ValueType>());

  std::vector<EmptyBlock>().swap(emptyBlocks);
  for (int i = 0; i <= kMaxLeafNum; i++) {
    emptyBlocks.push_back(EmptyBlock(i));
  }
}

template <typename KeyType, typename ValueType>
int CARMI<KeyType, ValueType>::AllocateSingleMemory(int *idx) {
  int newLeft = -1;
  for (int i = *idx; i < static_cast<int>(emptyBlocks.size()); i++) {
    newLeft = emptyBlocks[i].allocate();
    if (newLeft != -1) {
      *idx = i;
      break;
    }
  }
  return newLeft;
}

template <typename KeyType, typename ValueType>
int CARMI<KeyType, ValueType>::AllocateMemory(int neededLeafNumber) {
  int newLeft = -1;
  int idx = neededLeafNumber;
  newLeft = AllocateSingleMemory(&idx);
  // allocation fails, need to expand the entireData
  if (newLeft == -1) {
    entireData.resize(entireDataSize * 1.5, LeafSlots<KeyType, ValueType>());
    AllocateEmptyBlock(entireDataSize, entireDataSize * 0.5);
    entireDataSize *= 1.5;
    newLeft = AllocateSingleMemory(&idx);
  }

  // if the allocated size is less than block size, add the rest empty blocks
  // into the corresponding blocks
  if (neededLeafNumber < emptyBlocks[idx].m_width) {
    AllocateEmptyBlock(newLeft + neededLeafNumber,
                       emptyBlocks[idx].m_width - neededLeafNumber);
  }

  // update the right bound of data points in the entireData
  if (newLeft + neededLeafNumber > static_cast<int>(nowDataSize)) {
    nowDataSize = newLeft + neededLeafNumber;
  }
  return newLeft;
}

template <typename KeyType, typename ValueType>
void CARMI<KeyType, ValueType>::ReleaseMemory(int left, int size) {
  int len = size;
  int idx = 1;
  while (idx < 7) {
    if (emptyBlocks[idx].find(left + len)) {
      emptyBlocks[idx].m_block.erase(left + len);
      len += emptyBlocks[idx].m_width;
      idx = 1;
    } else {
      idx++;
    }
  }
  AllocateEmptyBlock(left, len);
}

#endif  // SRC_INCLUDE_DATAMANAGER_DATAPOINT_H_
