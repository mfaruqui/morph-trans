#include "sep-morph.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

double DROPOUT_RATE = 0.5;

SepMorph::SepMorph(const unsigned& char_length, const unsigned& hidden_length,
                       const unsigned& vocab_length, const unsigned& num_layers,
                       const unsigned& num_morph, Model *m) {
  char_len = char_length;
  hidden_len = hidden_length;
  vocab_len = vocab_length;
  layers = num_layers;
  morph_len = num_morph;
  InitParams(m);
}

void SepMorph::InitParams(Model *m) {
  for (unsigned i = 0; i < morph_len; ++i) {
    input_forward.push_back(LSTMBuilder(layers, char_len, hidden_len, m));
    input_backward.push_back(LSTMBuilder(layers, char_len, hidden_len, m));
    output_forward.push_back(LSTMBuilder(layers, 2 * char_len + hidden_len, hidden_len, m));

    phidden_to_output.push_back(m->add_parameters({vocab_len, hidden_len}));
    phidden_to_output_bias.push_back(m->add_parameters({vocab_len, 1}));

    char_vecs.push_back(m->add_lookup_parameters(vocab_len, {char_len}));

    ptransform_encoded.push_back(m->add_parameters({hidden_len, 2 * hidden_len}));
    ptransform_encoded_bias.push_back(m->add_parameters({hidden_len, 1}));

    peps_vec.push_back(m->add_parameters({char_len, 1}));
  }
}

void SepMorph::AddParamsToCG(const unsigned& morph_id, ComputationGraph* cg) {
  input_forward[morph_id].new_graph(*cg);
  input_backward[morph_id].new_graph(*cg);
  output_forward[morph_id].new_graph(*cg);

  hidden_to_output = parameter(*cg, phidden_to_output[morph_id]);
  hidden_to_output_bias = parameter(*cg, phidden_to_output_bias[morph_id]);

  transform_encoded = parameter(*cg, ptransform_encoded[morph_id]);
  transform_encoded_bias = parameter(*cg, ptransform_encoded_bias[morph_id]);
    
  EPS = parameter(*cg, peps_vec[morph_id]);
}

void SepMorph::RunFwdBwd(const unsigned& morph_id, const vector<unsigned>& inputs,
                           Expression* hidden, ComputationGraph *cg) {
  vector<Expression> input_vecs;
  for (const unsigned& input_id : inputs) {
    input_vecs.push_back(lookup(*cg, char_vecs[morph_id], input_id));
  }

  // Run forward LSTM
  Expression forward_unit;
  input_forward[morph_id].start_new_sequence();
  for (unsigned i = 0; i < input_vecs.size(); ++i) {
    forward_unit = input_forward[morph_id].add_input(input_vecs[i]);
  }

  // Run backward LSTM
  reverse(input_vecs.begin(), input_vecs.end());
  Expression backward_unit;
  input_backward[morph_id].start_new_sequence();
  for (unsigned i = 0; i < input_vecs.size(); ++i) {
    backward_unit = input_backward[morph_id].add_input(input_vecs[i]);
  }

  // Concatenate the forward and back hidden layers
  *hidden = concatenate({forward_unit, backward_unit});
}

void SepMorph::TransformEncodedInputForDecoding(Expression* encoded_input) const {
  *encoded_input = affine_transform({transform_encoded_bias,
                                     transform_encoded, *encoded_input});
}

void SepMorph::ProjectToOutput(const Expression& hidden, Expression* out) const {
  *out = affine_transform({hidden_to_output_bias, hidden_to_output, hidden});
}


Expression SepMorph::ComputeLoss(const vector<Expression>& hidden_units,
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

float SepMorph::Train(const unsigned& morph_id, const vector<unsigned>& inputs,
                        const vector<unsigned>& outputs,
                        AdadeltaTrainer* ada_gd) {
  ComputationGraph cg;
  AddParamsToCG(morph_id, &cg);

  // Encode and Transform to feed into decoder
  Expression encoded_input_vec;
  RunFwdBwd(morph_id, inputs, &encoded_input_vec, &cg);
  TransformEncodedInputForDecoding(&encoded_input_vec);

  // Use this encoded word vector to predict the transformed word
  vector<Expression> input_vecs_for_dec;
  vector<unsigned> output_ids_for_pred;
  for (unsigned i = 0; i < outputs.size(); ++i) {
    if (i < outputs.size() - 1) { 
      // '</s>' will not be fed as input -- it needs to be predicted.
      if (i < inputs.size() - 1) {
        input_vecs_for_dec.push_back(concatenate(
            {encoded_input_vec, lookup(cg, char_vecs[morph_id], outputs[i]),
             lookup(cg, char_vecs[morph_id], inputs[i + 1])}));
      } else {
        input_vecs_for_dec.push_back(concatenate(
            {encoded_input_vec, lookup(cg, char_vecs[morph_id], outputs[i]), EPS}));
      }
    }
    if (i > 0) {  // '<s>' will not be predicted in the output -- its fed in.
      output_ids_for_pred.push_back(outputs[i]);
    }
  }

  vector<Expression> decoder_hidden_units;
  output_forward[morph_id].start_new_sequence();
  for (const auto& vec : input_vecs_for_dec) {
    decoder_hidden_units.push_back(output_forward[morph_id].add_input(vec));
  }
  Expression loss = ComputeLoss(decoder_hidden_units, output_ids_for_pred);

  float return_loss = as_scalar(cg.forward());
  cg.backward();
  ada_gd->update(1.0f);
  return return_loss;
}
