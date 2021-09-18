/**
 * @file leaf_node_time.cpp
 * @author Jiaoyi
 * @brief
 * @version 0.1
 * @date 2021-09-18
 *
 * @copyright Copyright (c) 2021
 *
 */
#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "../include/carmi.h"
#include "../include/construct/minor_function.h"
#include "../include/dataManager/datapoint.h"
#include "../include/func/inlineFunction.h"
#include "../include/nodes/leafNode/array_type.h"

const int kSize = 110;
const float kSecondToNanosecond = 1000000000.0;
const int kModelNumber = 100000000;
const int block = 512;
const int end = kModelNumber / block;
std::vector<std::pair<double, double>> dataset(kSize);
std::vector<LeafSlots<double, double>> entireData;
std::vector<int> idx(end);

template <typename TYPE>
double GetNodePredictTime(CARMI<double, double> carmi) {
  std::vector<TYPE> node(kModelNumber);
  for (int i = 0; i < end; i++) {
    int neededLeafNum = carmi.GetActualSize(kSize);
    carmi.StoreData(neededLeafNum, 0, kSize, dataset, &node[i * block]);
  }
  std::vector<int> keys(kSize);
  for (int i = 0; i < kSize; i++) {
    keys[i] = i;
  }

  unsigned seed = std::clock();
  std::default_random_engine engine(seed);
  shuffle(idx.begin(), idx.end(), engine);
  shuffle(keys.begin(), keys.end(), engine);

  int tmpIdx, type, key;
  int currunion, find_idx;
  std::clock_t s, e;
  double tmp, tmp1 = 0;
  s = std::clock();
  for (int i = 0; i < end; i++) {
    tmpIdx = idx[i];
    key = keys[i % kSize];
    int left = node[tmpIdx].m_left;
    currunion = node[tmpIdx].Predict(key);
    find_idx += left + currunion;
  }
  e = std::clock();
  tmp = (e - s) / static_cast<double>(CLOCKS_PER_SEC);
  std::cout << "size:" << sizeof(node[0]) << "\n";
  s = std::clock();
  for (int i = 0; i < end; i++) {
    tmpIdx = idx[i];
    key = keys[i % kSize];
    find_idx += node[tmpIdx].m_left + node[tmpIdx].slotkeys[3];
  }
  e = std::clock();
  tmp1 = (e - s) / static_cast<double>(CLOCKS_PER_SEC);
  std::cout << "find idx:" << find_idx << std::endl;
  return (tmp - tmp1) * kSecondToNanosecond / end;
}

double GetBlockSearchTime(CARMI<double, double> carmi) {
  std::vector<LeafSlots<double, double>> tmpSlots(carmi.kMaxSlotNum,
                                                  LeafSlots<double, double>());
  for (int i = 0; i < carmi.kMaxSlotNum; i++) {
    for (int j = 0; j < i + 1; j++) {
      tmpSlots[i].slots[j] = {i, i * 10};
    }
  }
  unsigned seed = std::clock();
  std::default_random_engine engine(seed);
  for (int i = 0; i < kModelNumber; i++) {
    entireData.push_back(tmpSlots[i % carmi.kMaxSlotNum]);
  }
  std::vector<int> keys(carmi.kMaxSlotNum);
  for (int i = 0; i < carmi.kMaxSlotNum; i++) {
    keys[i] = i;
  }

  shuffle(idx.begin(), idx.end(), engine);
  shuffle(keys.begin(), keys.end(), engine);

  int tmpIdx, type, key;
  int currunion, find_idx, res;
  std::clock_t s, e;
  double tmp, tmp1 = 0;
  s = std::clock();
  for (int i = 0; i < end; i++) {
    tmpIdx = idx[i];
    key = keys[i % carmi.kMaxSlotNum];
    res += carmi.SlotsUnionSearch(entireData[tmpIdx], key);
  }
  e = std::clock();
  tmp = (e - s) / static_cast<double>(CLOCKS_PER_SEC);
  s = std::clock();
  for (int i = 0; i < end; i++) {
    tmpIdx = idx[i];
    key = keys[i % carmi.kMaxSlotNum];
    find_idx +=
        entireData[tmpIdx].slots[0].first + entireData[tmpIdx].slots[4].first +
        entireData[tmpIdx].slots[8].first + entireData[tmpIdx].slots[12].first;
  }
  e = std::clock();
  tmp1 = (e - s) / static_cast<double>(CLOCKS_PER_SEC);
  std::cout << "find idx:" << find_idx << std::endl;
  return (tmp - tmp1) * kSecondToNanosecond / end;
}

int main() {
  for (int i = 0; i < kSize; i++) {
    dataset[i] = {i, i * 10};
  }
  for (int i = 0; i < end; i++) {
    idx[i] = i * block;
  }
  CARMI<double, double> carmi;
  double cf = 0, block = 0;
  float times = 1.0;
  for (int i = 0; i < times; i++) {
    cf += GetNodePredictTime<ArrayType<double>>(carmi);
    block += GetBlockSearchTime(carmi);
  }

  std::cout << "cf average time:" << cf / times << std::endl;
  std::cout << "block average time:" << block / times << std::endl;
}
