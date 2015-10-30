#ifndef DECODE_H_
#define DECODE_H_

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/training.h"
#include "cnn/gpu-ops.h"
#include "cnn/expr.h"

#include "utils.h"
#include "ensemble.h"
#include "morph-trans.h"

#include <unordered_map>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

template<typename T> void
Decode(unordered_map<string, unsigned>& char_to_id,
       const vector<unsigned>& input_ids, vector<unsigned>* pred_target_ids,
       T* model);

template<typename T> void
EnsembleDecode(unordered_map<string, unsigned>& char_to_id,
               const vector<unsigned>& input_ids,
               vector<unsigned>* pred_target_ids, Ensemble<T>* ensmb_model);

#endif
