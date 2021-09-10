/**
 * @file construction.h
 * @author Jiaoyi
 * @brief main functions for CARMI
 * @version 0.1
 * @date 2021-03-11
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef SRC_INCLUDE_CONSTRUCT_CONSTRUCTION_H_
#define SRC_INCLUDE_CONSTRUCT_CONSTRUCTION_H_

#include <algorithm>
#include <iomanip>
#include <map>
#include <utility>
#include <vector>

#include "../carmi.h"
#include "./construct_root.h"
#include "./structures.h"

template <typename KeyType, typename ValueType>
inline void CARMI<KeyType, ValueType>::ConstructSubTree(
    const RootStruct &rootStruct, const SubDataset &subDataset,
    NodeCost *nodeCost) {
  float tmp = 0;
  int preI = 0;
  int checkpoint = -1;
  int dataNum = 0;
  int prefetchNum = 0;
  std::map<int, std::vector<double>> CostPerLeaf;

  for (int i = 0; i < rootStruct.rootChildNum; i++) {
    COST.insert({emptyRange, emptyCost});

    NodeCost resChild;
    DataRange range(subDataset.subInit[i], subDataset.subFind[i],
                    subDataset.subInsert[i]);

    if (subDataset.subInit[i].size > carmi_params::kAlgorithmThreshold)
      resChild = GreedyAlgorithm(range);
    else
      resChild = DP(range);

    // for prefetch
    auto it = structMap.find(range.initRange);
    if ((it->second.array.flagNumber >> 24) == ARRAY_LEAF_NODE &&
        range.initRange.left != -1) {
      int end = range.initRange.left + range.initRange.size;
      prefetchNum += range.initRange.size;
      int neededLeafNum = GetActualSize(range.initRange.size);

      if (i - preI > 10000) {
        checkpoint = prefetchData.size();
      }

      int avg = std::max(1.0, ceil(range.initRange.size * 1.0 / neededLeafNum));
      for (int j = range.initRange.left, k = 1; j < end; j++, k++) {
        double preIdx = root.model.PredictIdx(initDataset[j].first);
        prefetchData.push_back({preIdx, tmp});
        if (k == avg || j == end - 1) {
          k = 0;
          tmp++;
        }
        preI = i;
      }
      prefetchNode.push_back(i);
      prefetchRange.push_back(range);
      dataNum += range.initRange.size;
    } else {
      StoreOptimalNode(i, range);
    }

    nodeCost->cost += resChild.space * lambda + resChild.time;
    nodeCost->time += resChild.time;
    nodeCost->space += resChild.space;

    std::map<IndexPair, NodeCost>().swap(COST);
    std::map<IndexPair, BaseNode<KeyType>>().swap(structMap);
  }

  for (int i = 0; i < prefetchNode.size(); i++) {
    // calculate the cost of this array
    std::vector<double> cost(kMaxLeafNum, 0);
    for (int k = 0; k < kMaxLeafNum; k++) {
      double tmp_cost = CalculateArrayCost(prefetchRange[i].initRange.size,
                                           prefetchNum, k + 1);
      cost[k] = tmp_cost;
    }
    CostPerLeaf.insert({prefetchNode[i], cost});
  }
  if (!isPrimary) {
    int newBlockSize =
        root.fetch_model.PrefetchTrain(prefetchData, CostPerLeaf, checkpoint);
    entireData.resize(newBlockSize, LeafSlots<KeyType, ValueType>());
    entireDataSize = newBlockSize;
  }

  int largerThanGivenArray = 0;
  int largerThanGivenPoint = 0;

  for (int i = 0; i < prefetchNode.size(); i++) {
    ArrayType<KeyType> node;
    int neededLeafNum = GetActualSize(prefetchRange[i].initRange.size);
    int predictNum = root.fetch_model.PrefetchNum(prefetchNode[i]);
    bool isSuccess = false;
    if (neededLeafNum > predictNum) {
      largerThanGivenArray++;
      largerThanGivenPoint += prefetchRange[i].initRange.size;
    } else {
      isSuccess =
          StoreData(predictNum, prefetchRange[i].initRange.left,
                    prefetchRange[i].initRange.size, initDataset, &node);
    }
    if (!isSuccess) {
      remainingNode.push_back(prefetchNode[i]);
      remainingRange.push_back(prefetchRange[i]);
    } else {
      entireChild[prefetchNode[i]].array = node;
      scanLeaf.push_back(prefetchNode[i]);
    }
  }
  isInitMode = false;
  if (remainingNode.size() > 0) {
    for (int i = 0; i < remainingNode.size(); i++) {
      ArrayType<KeyType> node;
      int neededLeafNum = GetActualSize(remainingRange[i].initRange.size);
      auto isSuccess =
          StoreData(neededLeafNum, remainingRange[i].initRange.left,
                    remainingRange[i].initRange.size, initDataset, &node);
      entireChild[remainingNode[i]].array = node;
      scanLeaf.push_back(remainingNode[i]);
    }
  }

  std::vector<int>().swap(remainingNode);
  std::vector<DataRange>().swap(remainingRange);
}

template <typename KeyType, typename ValueType>
inline void CARMI<KeyType, ValueType>::Construction() {
  NodeCost nodeCost = emptyCost;
  RootStruct res = ChooseRoot();
  rootType = res.rootType;
  SubDataset subDataset = StoreRoot(res, &nodeCost);

#ifdef DEBUG
  std::cout << std::endl;
  std::cout << "constructing root is over!" << std::endl;
  std::cout << "the number of children is: " << res.rootChildNum << std::endl;
  time_t timep;
  time(&timep);
  char tmpTime[64];
  strftime(tmpTime, sizeof(tmpTime), "%Y-%m-%d %H:%M:%S", localtime(&timep));
  std::cout << "\nTEST time: " << tmpTime << std::endl;
#endif

  ConstructSubTree(res, subDataset, &nodeCost);
  UpdateLeaf();

  int neededSize = nowDataSize + reservedSpace;
  if (!isPrimary) {
    root.isPrefetch = true;

    if (neededSize < static_cast<int>(entireData.size())) {
      std::vector<LeafSlots<KeyType, ValueType>> tmpEntireData(
          entireData.begin(), entireData.begin() + neededSize);
      std::vector<LeafSlots<KeyType, ValueType>>().swap(entireData);
      entireData = tmpEntireData;
    }

    for (int i = 0; i < static_cast<int>(emptyBlocks.size()); i++) {
      auto it = emptyBlocks[i].m_block.lower_bound(neededSize);
      emptyBlocks[i].m_block.erase(it, emptyBlocks[i].m_block.end());
      auto tmp = emptyBlocks[i];
      for (auto j = tmp.m_block.begin(); j != tmp.m_block.end(); j++) {
        if (tmp.m_width + *j > static_cast<int>(entireData.size())) {
          AllocateEmptyBlock(*j, entireData.size() - *j);
          emptyBlocks[i].m_block.erase(*j);
          break;
        }
      }
    }
  }

  neededSize = nowChildNumber + reservedSpace;
  if (neededSize < static_cast<int>(entireChild.size())) {
    std::vector<BaseNode<KeyType>> tmp(entireChild.begin(),
                                       entireChild.begin() + neededSize);
    std::vector<BaseNode<KeyType>>().swap(entireChild);
    entireChild = tmp;
  }
  prefetchEnd = -1;
  DataVectorType().swap(initDataset);
  DataVectorType().swap(findQuery);
  DataVectorType().swap(insertQuery);
  std::vector<int>().swap(insertQueryIndex);
}

#endif  // SRC_INCLUDE_CONSTRUCT_CONSTRUCTION_H_
