#include "morph-trans.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

MorphTrans::MorphTrans(const unsigned& char_length, const unsigned& hidden_length,
                       const unsigned& vocab_length, const unsigned& num_layers, Model *m) {
  char_len = char_length;
  hidden_len = hidden_length;
  vocab_len = vocab_length;
  layers = num_layers;
  InitParams(m);
}

void MorphTrans::InitParams(Model *m) {
  input_forward = LSTMBuilder(layers, char_len, hidden_len, m);
  input_backward = LSTMBuilder(layers, char_len, hidden_len, m);
  output_forward = LSTMBuilder(layers, 2 * char_len + hidden_len, hidden_len, m);

  phidden_to_output = m->add_parameters({vocab_len, hidden_len});
  phidden_to_output_bias = m->add_parameters({vocab_len, 1});

  char_vecs = m->add_lookup_parameters(vocab_len, {char_len});

  ptransform_encoded = m->add_parameters({hidden_len, 2 * hidden_len});
  ptransform_encoded_bias = m->add_parameters({hidden_len, 1});

  peps_vec = m->add_parameters({char_len, 1});
}

void MorphTrans::AddParamsToCG(ComputationGraph* cg) {
  input_forward.new_graph(*cg);
  input_backward.new_graph(*cg);
  output_forward.new_graph(*cg);

  hidden_to_output = parameter(*cg, phidden_to_output);
  hidden_to_output_bias = parameter(*cg, phidden_to_output_bias);

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

void MorphTrans::ProjectToOutput(const Expression& hidden, Expression* out) const {
  *out = affine_transform({hidden_to_output_bias, hidden_to_output, hidden});
}


Expression MorphTrans::ComputeLoss(const vector<Expression>& hidden_units,
                                  const vector<unsigned>& targets) const {
  assert(hidden_units.size() == targets.size());
  vector<Expression> losses;
  for (unsigned i = 0; i < hidden_units.size(); ++i) {
    Expression out;
    ProjectToOutput(hidden_units[i], &out);
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
