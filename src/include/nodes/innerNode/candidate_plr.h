/**
 * @file candicate_plr.h
 * @author Jiaoyi
 * @brief
 * @version 0.1
 * @date 2021-03-16
 *
 * @copyright Copyright (c) 2021
 *
 */
#include <math.h>

#include <map>
#include <utility>
#include <vector>

#include "../../construct/structures.h"
#include "../../params.h"

#ifndef SRC_INCLUDE_NODES_ROOTNODE_TRAINMODEL_CANDIDATE_PLR_H_
#define SRC_INCLUDE_NODES_ROOTNODE_TRAINMODEL_CANDIDATE_PLR_H_

struct SegmentPoint {
  double cost;
  double key[12] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  int idx[12] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  int blockNum[12] = {1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1};
};

template <typename DataVectorType, typename DataType>
class CandidateCost {
 public:
  CandidateCost() {}

  void StoreTheta(const DataVectorType &dataset,
                  const std::vector<int> &index) {
    StoreValue(dataset, index);
    for (int i = 0; i < index.size() - 1; i++) {
      for (int j = i + 1; j < index.size(); j++) {
        int l = index[i];
        int r = index[j];
        auto tmp_theta = TrainLR(i, j, r - l);
        theta.insert({{l, r}, tmp_theta});
      }
    }
  }

  double Entropy(int leftIdx, int rightIdx) {
    auto tmp_theta = theta.find({leftIdx, rightIdx});
    double a = tmp_theta->second.first;
    double entropy = log2(a) * (rightIdx - leftIdx);
    return entropy;
  }

  std::pair<double, double> TrainLR(int left, int right, int size) {
    double theta1 = 0.0001, theta2 = 0.666;
    double t1 = 0, t2 = 0, t3 = 0, t4 = 0;
    t1 = xx[right] - xx[left];
    t2 = x[right] - x[left];
    t3 = px[right] - px[left];
    t4 = p[right] - p[left];

    theta1 = (t3 * size - t2 * t4) / (t1 * size - t2 * t2);
    theta2 = (t1 * t4 - t2 * t3) / (t1 * size - t2 * t2);

    return {theta1, theta2};
  }

  void StoreValue(const DataVectorType &dataset,
                  const std::vector<int> &index) {
    xx = std::vector<double>(index.size(), 0);
    x = std::vector<double>(index.size(), 0);
    px = std::vector<double>(index.size(), 0);
    p = std::vector<double>(index.size(), 0);
    xx[0] = 0.0;
    x[0] = 0.0;
    px[0] = 0.0;
    p[0] = 0.0;
    for (int i = 1; i < static_cast<int>(index.size()); i++) {
      for (int j = index[i - 1]; j < index[i]; j++) {
        xx[i] += dataset[j].first * dataset[j].first;
        x[i] += dataset[j].first;
        px[i] += dataset[j].first * dataset[j].second;
        p[i] += dataset[j].second;
      }
      xx[i] += xx[i - 1];
      x[i] += x[i - 1];
      px[i] += px[i - 1];
      p[i] += p[i - 1];
    }
  }

 public:
  std::map<std::pair<int, int>, std::pair<double, double>> theta;
  std::vector<double> xx;
  std::vector<double> x;
  std::vector<double> px;
  std::vector<double> p;
};

#endif  // SRC_INCLUDE_NODES_ROOTNODE_TRAINMODEL_CANDIDATE_PLR_H_
