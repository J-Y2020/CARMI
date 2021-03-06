/**
 * @file store_node.h
 * @author Jiaoyi
 * @brief store inner and leaf nodes
 * @version 0.1
 * @date 2021-03-11
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef SRC_INCLUDE_CONSTRUCT_STORE_NODE_H_
#define SRC_INCLUDE_CONSTRUCT_STORE_NODE_H_

#include <float.h>

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "../carmi.h"
#include "../dataManager/child_array.h"
#include "../dataManager/datapoint.h"
#include "../nodes/innerNode/bs_model.h"
#include "../nodes/innerNode/his_model.h"
#include "../nodes/innerNode/lr_model.h"
#include "../nodes/innerNode/plr_model.h"
#include "../nodes/leafNode/array_type.h"
#include "../nodes/leafNode/external_array_type.h"
#include "../params.h"
#include "./dp_inner.h"

template <typename KeyType, typename ValueType>
template <typename TYPE>
TYPE CARMI<KeyType, ValueType>::StoreInnerNode(const IndexPair &range,
                                               TYPE *node) {
  int optimalChildNumber = node->flagNumber & 0x00FFFFFF;
  SubDataset subDataset(optimalChildNumber);

  NodePartition<TYPE>(*node, range, initDataset, &(subDataset.subInit));
  node->childLeft = AllocateChildMemory(optimalChildNumber);

  for (int i = 0; i < optimalChildNumber; i++) {
    DataRange subRange(subDataset.subInit[i], subDataset.subFind[i],
                       subDataset.subInsert[i]);
    StoreOptimalNode(node->childLeft + i, subRange);
  }
  return *node;
}

template <typename KeyType, typename ValueType>
void CARMI<KeyType, ValueType>::StoreOptimalNode(int storeIdx,
                                                 const DataRange &range) {
  auto it = structMap.find(range.initRange);

  int type = it->second.array.flagNumber >> 24;
  switch (type) {
    case LR_INNER_NODE: {
      StoreInnerNode<LRModel>(range.initRange, &(it->second.lr));
      entireChild[storeIdx].lr = it->second.lr;
      break;
    }
    case PLR_INNER_NODE: {
      StoreInnerNode<PLRModel>(range.initRange, &(it->second.plr));
      entireChild[storeIdx].plr = it->second.plr;
      break;
    }
    case HIS_INNER_NODE: {
      StoreInnerNode<HisModel>(range.initRange, &(it->second.his));
      entireChild[storeIdx].his = it->second.his;
      break;
    }
    case BS_INNER_NODE: {
      StoreInnerNode<BSModel>(range.initRange, &(it->second.bs));
      entireChild[storeIdx].bs = it->second.bs;
      break;
    }
    case ARRAY_LEAF_NODE: {
      int neededLeafNum = GetActualSize(range.initRange.size);
      remainingNode.push_back(storeIdx);
      remainingRange.push_back(range);
      break;
    }
    case EXTERNAL_ARRAY_LEAF_NODE: {
      ExternalArray node = it->second.externalArray;
      int size = range.initRange.size;
      if (size <= 0)
        node.m_left = curr;
      else
        node.m_left = range.initRange.left;
      entireChild[storeIdx].externalArray = node;
      break;
    }
  }
  if (type >= ARRAY_LEAF_NODE && firstLeaf == -1) {
    firstLeaf = storeIdx;
  }
}

#endif  // SRC_INCLUDE_CONSTRUCT_STORE_NODE_H_
