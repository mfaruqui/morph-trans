#ifndef JOINT_ENC_MORPH_H_
#define JOINT_ENC_MORPH_H_

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

// The class has a shared encoder acorss all morphological types.
// Other auxiliary units are also shared, except for the decoder and
// input transofrmation parameters.
class JointEncMorph {
 public:
  LSTMBuilder input_forward, input_backward;  // Shared encoder
  vector<LSTMBuilder> output_forward;
  LookupParameters* char_vecs;  // Shared char vectors

  Expression hidden_to_output, hidden_to_output_bias;
  Parameters *phidden_to_output, *phidden_to_output_bias;  // Shared

  Expression transform_encoded, transform_encoded_bias;
  vector<Parameters*> ptransform_encoded, ptransform_encoded_bias;

  vector<LookupParameters*> eps_vecs; // Not sharing the epsilon vectors
  
  unsigned char_len, hidden_len, vocab_len, layers, morph_len, max_eps = 5;

  JointEncMorph() {}

  JointEncMorph(const unsigned& char_length, const unsigned& hidden_length,
           const unsigned& vocab_length, const unsigned& layers,
           const unsigned& num_morph, vector<Model*>* m,
           vector<AdadeltaTrainer>* optimizer);

  void InitParams(vector<Model*>* m);

  void AddParamsToCG(const unsigned& morph_id, ComputationGraph* cg);

  void RunFwdBwd(const vector<unsigned>& inputs,
                 Expression* hidden, ComputationGraph *cg);

  void TransformEncodedInput(Expression* encoded_input) const; 

  void ProjectToOutput(const Expression& hidden, Expression* out) const;

  Expression ComputeLoss(const vector<Expression>& hidden_units,
                         const vector<unsigned>& targets) const;

  float Train(const unsigned& morph_id, const vector<unsigned>& inputs,
              const vector<unsigned>& outputs, AdadeltaTrainer* opt,
              AdadeltaTrainer* shared_opt);

  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    ar & char_len;
    ar & hidden_len;
    ar & vocab_len;
    ar & layers;
    ar & morph_len;
    ar & max_eps;
  }
};

void Serialize(string& filename, JointEncMorph& model, vector<Model*>* cnn_model);

void Read(string& filename, JointEncMorph* model, vector<Model*>* cnn_model);

void
EnsembleDecode(const unsigned& morph_id, unordered_map<string, unsigned>& char_to_id,
               const vector<unsigned>& input_ids, vector<unsigned>* pred_target_ids,
               vector<JointEncMorph*>* ensmb_model);

void
EnsembleBeamDecode(const unsigned& morph_id, const unsigned& beam_size,
                   unordered_map<string, unsigned>& char_to_id,
                   const vector<unsigned>& input_ids,
                   vector<vector<unsigned> >* sequences, vector<float>* tm_scores,
                   vector<JointEncMorph*>* ensmb_model);

#endif
