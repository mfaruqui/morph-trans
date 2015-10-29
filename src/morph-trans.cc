#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/training.h"
#include "cnn/gpu-ops.h"
#include "cnn/expr.h"

#include "utils.h"
#include "morph-trans.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

string BOW = "<s>", EOW = "</s>";
unsigned MAX_PRED_LEN = 100;

MorphTrans::MorphTrans(const int& char_length, const int& hidden_length,
                       const int& vocab_length, const int& layers, Model *m) {
  char_len = char_length;

  input_forward = LSTMBuilder(layers, char_length, hidden_length, m);
  input_backward = LSTMBuilder(layers, char_length, hidden_length, m);
  output_forward = LSTMBuilder(layers, 2 * char_length + hidden_length, hidden_length, m);

  proj_to_vocab = ProjToOutput(hidden_length, vocab_length, m);

  char_vecs = m->add_lookup_parameters(vocab_length, {char_length});

  ptransform_encoded = m->add_parameters({hidden_length, 2 * hidden_length});
  ptransform_encoded_bias = m->add_parameters({hidden_length, 1});
    
  peps_vec = m->add_parameters({char_len, 1});
}

void MorphTrans::AddParamsToCG(ComputationGraph* cg) {
  input_forward.new_graph(*cg);
  input_backward.new_graph(*cg);
  output_forward.new_graph(*cg);

  proj_to_vocab.AddParamsToCG(cg);

  transform_encoded = parameter(*cg, ptransform_encoded);
  transform_encoded_bias = parameter(*cg, ptransform_encoded_bias);
    
  EPS = parameter(*cg, peps_vec);
}

void MorphTrans::RunFwdBwd(const vector<unsigned>& inputs,
                           Expression* hidden, ComputationGraph *cg) {
  vector<Expression> input_vecs;
  for (const unsigned& input_id : inputs) {
    input_vecs.push_back(lookup(*cg, char_vecs, input_id));
  }

  // Run forward LSTM
  Expression forward_unit;
  input_forward.start_new_sequence();
  for (unsigned i = 0; i < input_vecs.size(); ++i) {
    forward_unit = input_forward.add_input(input_vecs[i]);
  }

  // Run backward LSTM
  reverse(input_vecs.begin(), input_vecs.end());
  Expression backward_unit;
  input_backward.start_new_sequence();
  for (unsigned i = 0; i < input_vecs.size(); ++i) {
    backward_unit = input_backward.add_input(input_vecs[i]);
  }

  // Concatenate the forward and back hidden layers
  *hidden = concatenate({forward_unit, backward_unit});
}

void MorphTrans::TransformEncodedInputForDecoding(Expression* encoded_input) const {
  *encoded_input = affine_transform({transform_encoded_bias,
                                     transform_encoded, *encoded_input});
}

Expression MorphTrans::ComputeLoss(const vector<Expression>& hidden_units,
                                  const vector<unsigned>& targets) const {
  assert(hidden_units.size() == targets.size());
  vector<Expression> losses;
  for (unsigned i = 0; i < hidden_units.size(); ++i) {
    Expression out;
    proj_to_vocab.ProjectToOutput(hidden_units[i], &out);
    losses.push_back(pickneglogsoftmax(out, targets[i]));
  }
  return sum(losses);
}

float MorphTrans::Train(const vector<unsigned>& inputs,
                        const vector<unsigned>& outputs,
                        AdadeltaTrainer* ada_gd) {
  ComputationGraph cg;
  AddParamsToCG(&cg);

  // Encode and Transform to feed into decoder
  Expression encoded_input_vec;
  RunFwdBwd(inputs, &encoded_input_vec, &cg);
  TransformEncodedInputForDecoding(&encoded_input_vec);

  // Use this encoded word vector to predict the transformed word
  vector<Expression> input_vecs_for_dec;
  vector<unsigned> output_ids_for_pred;
  for (unsigned i = 0; i < outputs.size(); ++i) {
    if (i < outputs.size() - 1) { 
      // '</s>' will not be fed as input -- it needs to be predicted.
      if (i < inputs.size() - 1) {
        input_vecs_for_dec.push_back(concatenate(
            {encoded_input_vec, lookup(cg, char_vecs, outputs[i]),
             lookup(cg, char_vecs, inputs[i+1])}));
      } else {
        input_vecs_for_dec.push_back(concatenate(
            {encoded_input_vec, lookup(cg, char_vecs, outputs[i]), EPS}));
      }
    }
    if (i > 0) {  // '<s>' will not be predicted in the output -- its fed in.
      output_ids_for_pred.push_back(outputs[i]);
    }
  }

  vector<Expression> decoder_hidden_units;
  output_forward.start_new_sequence();
  for (const auto& vec : input_vecs_for_dec) {
    decoder_hidden_units.push_back(output_forward.add_input(vec));
  }
  Expression loss = ComputeLoss(decoder_hidden_units, output_ids_for_pred);

  float return_loss = as_scalar(cg.forward());
  cg.backward();
  ada_gd->update(1.0f);
  return return_loss;
}

void MorphTrans::Decode(const Expression& encoded_word_vec,
                        unordered_map<string, unsigned>& char_to_id,
                        const vector<unsigned>& input_ids,
                        vector<unsigned>* pred_target_ids,
                        ComputationGraph* cg) {
  Expression input_word_vec = lookup(*cg, char_vecs, char_to_id[BOW]);
  pred_target_ids->push_back(char_to_id[BOW]);
  output_forward.start_new_sequence();
  unsigned out_index = 1;
  while (pred_target_ids->size() < MAX_PRED_LEN) {
    Expression input;
    if (out_index < input_ids.size()) {
      input = concatenate({encoded_word_vec, input_word_vec,
                          lookup(*cg, char_vecs, input_ids[out_index])});
    } else {
      input = concatenate({encoded_word_vec, input_word_vec, EPS});
    }
    Expression hidden = output_forward.add_input(input);
    Expression out;
    proj_to_vocab.ProjectToOutput(hidden, &out);
    vector<float> dist = as_vector(cg->incremental_forward());
    unsigned pred_index = 0;
    float best_score = dist[pred_index];
    for (unsigned index = 1; index < dist.size(); ++index) {
      if (dist[index] > best_score) {
        best_score = dist[index];
        pred_index = index;
      }
    }
    pred_target_ids->push_back(pred_index);
    if (pred_index == char_to_id[EOW]) {
      return;  // If the end is found, break from the loop and return
    }
    input_word_vec = lookup(*cg, char_vecs, pred_index);
    out_index++;
  }
}

void MorphTrans::Predict(const vector<unsigned>& inputs,
                         unordered_map<string, unsigned>& char_to_id,
                         vector<unsigned>* outputs) {
  ComputationGraph cg;
  AddParamsToCG(&cg);

  // Encode and Transofrm to feed into decoder
  Expression encoded_input_vec;
  RunFwdBwd(inputs, &encoded_input_vec, &cg);
  TransformEncodedInputForDecoding(&encoded_input_vec);

  // Make preditions using the decoder.
  Decode(encoded_input_vec, char_to_id, inputs, outputs, &cg);
}
