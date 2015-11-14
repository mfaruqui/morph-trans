#ifndef INFL_TO_ROOT_H_
#define INFL_TO_ROOT_H_

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
#include <unordered_map>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

// Reduces inflected form to root.
class InflToRoot {
 public:
  LSTMBuilder input_forward, input_backward, output_forward;  // Shared encoder
  LookupParameters* char_vecs;  // Shared char vectors

  Expression hidden_to_output, hidden_to_output_bias;
  Parameters *phidden_to_output, *phidden_to_output_bias;  // Shared

  Expression transform_encoded, transform_encoded_bias;
  Parameters *ptransform_encoded, *ptransform_encoded_bias;

  LookupParameters* eps_vecs;
  
  unsigned char_len, hidden_len, vocab_len, layers, max_eps = 5;

  InflToRoot() {}

  InflToRoot(const unsigned& char_length, const unsigned& hidden_length,
             const unsigned& vocab_length, const unsigned& layers,
             Model* m, AdadeltaTrainer* optimizer);

  void InitParams(Model* m);

  void AddParamsToCG(ComputationGraph* cg);

  void RunFwdBwd(const vector<unsigned>& inputs,
                 Expression* hidden, ComputationGraph *cg);

  void TransformEncodedInput(Expression* encoded_input) const;

  void ProjectToOutput(const Expression& hidden, Expression* out) const;

  Expression ComputeLoss(const vector<Expression>& hidden_units,
                         const vector<unsigned>& targets) const;

  float Train(const vector<unsigned>& inputs,
              const vector<unsigned>& outputs, AdadeltaTrainer* opt);

  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    ar & char_len;
    ar & hidden_len;
    ar & vocab_len;
    ar & layers;
    ar & max_eps;
  }
};

void Serialize(string& filename, InflToRoot& model, Model* cnn_model);

void Read(string& filename, InflToRoot* model, Model* cnn_model);

void
EnsembleDecode(unordered_map<string, unsigned>& char_to_id,
               const vector<unsigned>& input_ids, vector<unsigned>* pred_target_ids,
               vector<InflToRoot*>* ensmb_model);

#endif
