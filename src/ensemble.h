#ifndef ENSEMBLE_H_
#define ENSEMBLE_H_

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/training.h"
#include "cnn/gpu-ops.h"
#include "cnn/expr.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <limits>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

template<typename T> class Ensemble {
 public:
  vector<T> models;
  int ensmb;

  Ensemble(const int& char_length, const int& hidden_length,
           const int& vocab_length, const int& layers, const int& ensmb_size,
           Model *m);

  void AddParamsToCG(ComputationGraph* cg);

  float Train(const vector<unsigned>& inputs, const vector<unsigned>& outputs,
              AdadeltaTrainer* ada_gd);
};

#endif  // ENSEMBLE_H_
