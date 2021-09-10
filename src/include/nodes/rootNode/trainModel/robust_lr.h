/**
 * @file robust_lr.h
 * @author Jiaoyi
 * @brief
 * @version 0.1
 * @date 2021-06-29
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef SRC_INCLUDE_NODES_ROOTNODE_TRAINMODEL_ROBUST_LR_H_
#define SRC_INCLUDE_NODES_ROOTNODE_TRAINMODEL_ROBUST_LR_H_

#include <float.h>
#include <math.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "../../../params.h"

template <typename DataVectorType, typename KeyType>
class RobustLR {
 public:
  RobustLR() {
    theta1 = 0.0001;
    theta2 = 0.666;
  }
  void Train(const DataVectorType &dataset, int len) {
    length = len - 1;
    int candiSize = 10;
    double maxDistance = 10.0, maxEntropy = 0.0, avgSize = 0.0;
    std::vector<int> candidateIndex(candiSize);
    int size = dataset.size();
    std::vector<double> index(size, 0);
    for (int idx = 0, i = 0; i < size; i++) {
      index[idx++] = static_cast<double>(i) / size * length;
    }
    for (int i = 0; i < candiSize; i++) {
      candidateIndex[i] = size / candiSize * i;
    }
    double optTheta1, optTheta2;
    for (int i = 0; i < candiSize; i++) {
      for (int j = 0; j < candiSize; j++) {
        double t1 =
            (index[j] - index[i]) / (dataset[j].first - dataset[i].first);
        double t2 = index[j] - t1 * dataset[i].first;

        DataVectorType subDataset;
        for (int k = 0; k < size; k++) {
          double dis = CalDistance({dataset[k].first, index[k]}, t1, t2);
          if (dis <= maxDistance) {
            subDataset.push_back({dataset[k].first, index[k]});
          }
        }

        LSMTrain(subDataset, len);
        std::vector<int> perSize(len);
        for (int k = 0; k < size; k++) {
          int p = Predict(dataset[k].first);
          perSize[p]++;
        }

        double avg = 0.0;
        double entropy = CalEntropy(perSize, size, &avg);
        avg = avg * 1.0 / len;

        if (entropy >= maxEntropy) {
          maxEntropy = entropy;
          optTheta1 = theta1;
          optTheta2 = theta2;
          avgSize = avg;
        }
      }
    }

    theta1 = optTheta1;
    theta2 = optTheta2;
  }

  void LSMTrain(const DataVectorType &dataset, int len) {
    int size = dataset.size();
    if (size == 0) return;

    double t1 = 0, t2 = 0, t3 = 0, t4 = 0;
    for (int i = 0; i < size; i++) {
      t1 += dataset[i].first * dataset[i].first;
      t2 += dataset[i].first;
      t3 += dataset[i].first * dataset[i].second;
      t4 += dataset[i].second;
    }
    theta1 = (t3 * size - t2 * t4) / (t1 * size - t2 * t2);
    theta2 = (t1 * t4 - t2 * t3) / (t1 * size - t2 * t2);
  }

  double CalDistance(const std::pair<KeyType, int> &p0, double theta1,
                     double theta2) {
    double res = 0.0;
    res = abs(p0.second - theta1 * p0.first - theta2) / (1.0 + theta1 * theta1);
    return res;
  }

  double CalEntropy(const std::vector<int> &perSize, int totalSize,
                    double *avg) {
    double entropy = 0.0;
    *avg = 0.0;
    int size = perSize.size();
    if (size == 0) {
      return DBL_MAX;
    }
    for (int i = 0; i < size; i++) {
      auto p = static_cast<float>(perSize[i]) / totalSize;
      *avg += perSize[i];
      if (p != 0) {
        entropy += p * (-log2(p));
      }
    }
    return entropy;
  }

  int Predict(KeyType key) const {
    // return the predicted idx in the children
    int p = theta1 * key + theta2;
    if (p < 0)
      p = 0;
    else if (p > length)
      p = length;
    return p;
  }

 private:
  int length;
  double theta1;
  double theta2;
};

#endif  // SRC_INCLUDE_NODES_ROOTNODE_TRAINMODEL_ROBUST_LR_H_
