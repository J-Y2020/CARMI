/**
 * @file lr_type.h
 * @author Jiaoyi
 * @brief
 * @version 0.1
 * @date 2021-03-11
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef SRC_INCLUDE_NODES_ROOTNODE_ROOT_NODES_H_
#define SRC_INCLUDE_NODES_ROOTNODE_ROOT_NODES_H_

#include <fstream>
#include <vector>

#include "../../construct/structures.h"
#include "../../params.h"
#include "trainModel/linear_regression.h"
#include "trainModel/piecewireLR.h"
#include "trainModel/prefetch_plr.h"
#include "trainModel/robust_lr.h"

template <typename DataVectorType, typename KeyType>
class PLRType {
 public:
  PLRType() = default;
  explicit PLRType(int c) {
    flagNumber = (PLR_ROOT_NODE << 24);
    childNumber = c;
    isPrefetch = false;
  }
  PiecewiseLR<DataVectorType, KeyType> model;
  PrefetchPLR<DataVectorType, KeyType> fetch_model;  // 20 Byte
  int flagNumber;
  int childNumber;
  int childLeft;
  bool isPrefetch;
};
#endif  // SRC_INCLUDE_NODES_ROOTNODE_ROOT_NODES_H_
