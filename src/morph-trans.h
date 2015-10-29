#ifndef MORPH_CHAR_H_
#define MORPH_CHAR_H_

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/training.h"
#include "cnn/gpu-ops.h"
#include "cnn/expr.h"

#include "utils.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <fstream>
#include <unordered_map>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

class MorphTrans {
 public:
  LSTMBuilder input_forward, input_backward, output_forward;
  LookupParameters* char_vecs;

  Expression hidden_to_output, hidden_to_output_bias;
  Parameters *phidden_to_output, *phidden_to_output_bias;

  Expression transform_encoded, transform_encoded_bias;
  Parameters *ptransform_encoded, *ptransform_encoded_bias;
  
  unsigned char_len;
  Expression ZERO;
  Parameters *pzero_vec;

  MorphTrans(const int& char_length, const int& hidden_length,
             const int& vocab_length, const int& layers, Model *m);

  void AddParamsToCG(ComputationGraph* cg);

  void RunFwdBwd(const vector<unsigned>& inputs,
                 Expression* hidden, ComputationGraph *cg);

  void TransformEncodedInputForDecoding(Expression* encoded_input) const;

  void ProjectToVocab(const Expression& hidden, Expression* out) const;

  Expression ComputeLoss(const vector<Expression>& hidden_units,
                         const vector<unsigned>& targets) const;

  float Train(const vector<unsigned>& inputs, const vector<unsigned>& outputs,
              AdadeltaTrainer* ada_gd);

  void Decode(const Expression& encoded_word_vec,
              unordered_map<string, unsigned>& char_to_id,
              vector<unsigned>* pred_target_ids,
              const vector<unsigned> input_ids, ComputationGraph* cg);

  void Predict(const vector<unsigned>& inputs,
               unordered_map<string, unsigned>& char_to_id,
               vector<unsigned>* outputs);
};

#endif
