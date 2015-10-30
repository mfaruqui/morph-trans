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
  
  unsigned char_len, hidden_len, vocab_len, layers;
  Expression EPS;
  Parameters *peps_vec;

  MorphTrans() {}

  MorphTrans(const unsigned& char_length, const unsigned& hidden_length,
             const unsigned& vocab_length, const unsigned& layers, Model *m);

  void InitParams(Model *m);

  void AddParamsToCG(ComputationGraph* cg);

  void RunFwdBwd(const vector<unsigned>& inputs,
                 Expression* hidden, ComputationGraph *cg);

  void TransformEncodedInputForDecoding(Expression* encoded_input) const;

  void ProjectToOutput(const Expression& hidden, Expression* out) const;

  Expression ComputeLoss(const vector<Expression>& hidden_units,
                         const vector<unsigned>& targets) const;

  float Train(const vector<unsigned>& inputs, const vector<unsigned>& outputs,
              AdadeltaTrainer* ada_gd);

  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    ar & char_len;
    ar & hidden_len;
    ar & vocab_len;
    ar & layers;
  }
};

#endif
