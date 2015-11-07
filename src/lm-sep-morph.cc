#include "lm-sep-morph.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;
using cnn::expr::input;

string BOW = "<s>", EOW = "</s>";
int MAX_PRED_LEN = 100;

LMSepMorph::LMSepMorph(const unsigned& char_length, const unsigned& hidden_length,
                   const unsigned& vocab_length, const unsigned& num_layers,
                   const unsigned& num_morph, vector<Model*>* m,
                   vector<AdadeltaTrainer>* optimizer) {
  char_len = char_length;
  hidden_len = hidden_length;
  vocab_len = vocab_length;
  layers = num_layers;
  morph_len = num_morph;
  InitParams(m);
}

void LMSepMorph::InitParams(vector<Model*>* m) {
  for (unsigned i = 0; i < morph_len; ++i) {
    input_forward.push_back(LSTMBuilder(layers, char_len, hidden_len, (*m)[i]));
    input_backward.push_back(LSTMBuilder(layers, char_len, hidden_len, (*m)[i]));
    output_forward.push_back(LSTMBuilder(layers, 2 * char_len + hidden_len,
                                         hidden_len, (*m)[i]));

    phidden_to_output.push_back((*m)[i]->add_parameters({vocab_len, hidden_len}));
    phidden_to_output_bias.push_back((*m)[i]->add_parameters({vocab_len, 1}));

    char_vecs.push_back((*m)[i]->add_lookup_parameters(vocab_len, {char_len}));

    ptransform_encoded.push_back((*m)[i]->add_parameters({hidden_len,
                                                          2 * hidden_len}));
    ptransform_encoded_bias.push_back((*m)[i]->add_parameters({hidden_len, 1}));

    eps_vecs.push_back((*m)[i]->add_lookup_parameters(max_eps, {char_len}));
    lm_pos_weights.push_back((*m)[i]->add_lookup_parameters(
      max_lm_pos_weights, {vocab_len}));
  }
}

void LMSepMorph::AddParamsToCG(const unsigned& morph_id, ComputationGraph* cg) {
  input_forward[morph_id].new_graph(*cg);
  input_backward[morph_id].new_graph(*cg);
  output_forward[morph_id].new_graph(*cg);

  hidden_to_output = parameter(*cg, phidden_to_output[morph_id]);
  hidden_to_output_bias = parameter(*cg, phidden_to_output_bias[morph_id]);

  transform_encoded = parameter(*cg, ptransform_encoded[morph_id]);
  transform_encoded_bias = parameter(*cg, ptransform_encoded_bias[morph_id]);
}

void LMSepMorph::RunFwdBwd(const unsigned& morph_id,
                         const vector<unsigned>& inputs,
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
  Expression backward_unit;
  input_backward[morph_id].start_new_sequence();
  for (int i = input_vecs.size() - 1; i >= 0; --i) {
    backward_unit = input_backward[morph_id].add_input(input_vecs[i]);
  }

  // Concatenate the forward and back hidden layers
  *hidden = concatenate({forward_unit, backward_unit});
}

void LMSepMorph::TransformEncodedInput(Expression* encoded_input) const {
  *encoded_input = affine_transform({transform_encoded_bias,
                                     transform_encoded, *encoded_input});
}

void LMSepMorph::ProjectToOutput(const Expression& hidden, Expression* out) const {
  *out = affine_transform({hidden_to_output_bias, hidden_to_output, hidden});
}


Expression LMSepMorph::ComputeLoss(const unsigned& morph_id,
                                   const vector<Expression>& hidden_units,
                                   const vector<unsigned>& targets,
                                   LM *lm, ComputationGraph* cg) const {
  vector<Expression> losses;
  vector<unsigned> incremental_targets;
  incremental_targets.push_back(lm->char_to_id[BOW]);
  for (unsigned i = 0; i < hidden_units.size(); ++i) {
    Expression out;
    ProjectToOutput(hidden_units[i], &out);
    Expression trans_lp = log_softmax(out);

    // Calculate the LM probabilities of all possible outputs.
    Expression lm_lp = LogProbDist(incremental_targets, lm, cg);

    unsigned lm_index = min(i + 1, max_lm_pos_weights - 1);
    Expression lm_weight = lookup(*cg, lm_pos_weights[morph_id], lm_index);

    Expression total_lp = trans_lp + cwise_multiply(lm_lp, Softplus(lm_weight));
    losses.push_back(pickneglogsoftmax(total_lp, targets[i]));
    incremental_targets.push_back(targets[i]);
  }
  return sum(losses);
}

float LMSepMorph::Train(const unsigned& morph_id,
                        const vector<unsigned>& inputs,
                        const vector<unsigned>& outputs, LM *lm,
                        AdadeltaTrainer* ada_gd) {
  ComputationGraph cg;
  AddParamsToCG(morph_id, &cg);

  // Encode and Transform to feed into decoder
  Expression encoded_input_vec;
  RunFwdBwd(morph_id, inputs, &encoded_input_vec, &cg);
  TransformEncodedInput(&encoded_input_vec);

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
            {encoded_input_vec, lookup(cg, char_vecs[morph_id], outputs[i]),
             lookup(cg, eps_vecs[morph_id], min(unsigned(i - inputs.size()), max_eps - 1))}));
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

  // If its the first iteration, do not use language model.
  Expression loss = ComputeLoss(morph_id, decoder_hidden_units,
                                output_ids_for_pred, lm, &cg);

  float return_loss = as_scalar(cg.incremental_forward());
  cg.backward();
  ada_gd->update(1.0f);

  cout << morph_id << " ";
  for (unsigned i = 1; i < 10; ++i) {
    float p = lm_pos_weights[morph_id]->values[i].v[8];
    cout << Softplus(p) << " ";
  }
  cout << endl;

  return return_loss;
}

Expression LogProbDist(const vector<unsigned>& seq, LM *lm,
                       ComputationGraph *cg) {
  vector<float> lm_dist(lm->char_to_id.size(), 0.f);
  for (const auto& it : lm->char_to_id) {
    vector<unsigned> possible_seq(seq);
    possible_seq.push_back(it.second);
    lm_dist[it.second] = lm->LogProbSeq(possible_seq);
  }
  return input(*cg, {(long) lm_dist.size()}, lm_dist);
}

void
EnsembleDecode(const unsigned& morph_id, unordered_map<string, unsigned>& char_to_id,
               const vector<unsigned>& input_ids, vector<unsigned>* pred_target_ids,
               LM* lm, vector<LMSepMorph*>* ensmb_model) {
  ComputationGraph cg;

  unsigned ensmb = ensmb_model->size();
  vector<Expression> encoded_word_vecs;
  for (unsigned i = 0; i < ensmb; ++i) {
    Expression encoded_word_vec;
    auto model = (*ensmb_model)[i];
    model->AddParamsToCG(morph_id, &cg);
    model->RunFwdBwd(morph_id, input_ids, &encoded_word_vec, &cg);
    model->TransformEncodedInput(&encoded_word_vec);
    encoded_word_vecs.push_back(encoded_word_vec);
    model->output_forward[morph_id].start_new_sequence();
  }

  unsigned out_index = 1;
  unsigned pred_index = char_to_id[BOW];
  while (pred_target_ids->size() < MAX_PRED_LEN) {
    vector<Expression> ensmb_out;
    pred_target_ids->push_back(pred_index);
    if (pred_index == char_to_id[EOW]) {
      return;  // If the end is found, break from the loop and return
    }

    vector<Expression> lm_weights;
    for (unsigned ensmb_id = 0; ensmb_id < ensmb; ++ensmb_id) {
      auto model = (*ensmb_model)[ensmb_id];
      Expression prev_output_vec = lookup(cg, model->char_vecs[morph_id], pred_index);
      Expression input, input_char_vec;
      if (out_index < input_ids.size()) {
        input_char_vec = lookup(cg, model->char_vecs[morph_id], input_ids[out_index]);
      } else {
        input_char_vec = lookup(cg, model->eps_vecs[morph_id],
                                min(unsigned(out_index - input_ids.size()),
                                             model->max_eps - 1));
      }
      input = concatenate({encoded_word_vecs[ensmb_id], prev_output_vec,
                           input_char_vec});

      Expression hidden = model->output_forward[morph_id].add_input(input);
      Expression out;
      model->ProjectToOutput(hidden, &out);
      ensmb_out.push_back(log_softmax(out));
      
      unsigned lm_index = min(out_index, model->max_lm_pos_weights - 1);
      lm_weights.push_back(lookup(cg, model->lm_pos_weights[morph_id], lm_index));
    }
    Expression trans_morph_dist = average(ensmb_out);
    Expression avg_lm_weight = average(lm_weights);

    // Compute the language model distribution now.
    Expression lm_dist = LogProbDist(*pred_target_ids, lm, &cg);
    
    // Sum the LM dist and the trans morph dist.
    Expression total_dist = trans_morph_dist +
                            cwise_multiply(lm_dist, Softplus(avg_lm_weight));

    vector<float> dist = as_vector(cg.incremental_forward());
    pred_index = distance(dist.begin(), max_element(dist.begin(), dist.end()));
    out_index++;
  }
}

float Softplus(float x) {
  return log(1 + exp(x));
}

Expression Softplus(Expression x) {
  return log(1 + exp(x));
}

void Serialize(string& filename, LMSepMorph& model, vector<Model*>* cnn_models) {
  ofstream outfile(filename);
  if (!outfile.is_open()) {
    cerr << "File opening failed" << endl;
  }

  boost::archive::text_oarchive oa(outfile);
  oa & model;
  for (unsigned i = 0; i < cnn_models->size(); ++i) {
    oa & *(*cnn_models)[i];
  }

  cerr << "Saved model to: " << filename << endl;
  outfile.close();
}

void Read(string& filename, LMSepMorph* model, vector<Model*>* cnn_models) {
  ifstream infile(filename);
  if (!infile.is_open()) {
    cerr << "File opening failed" << endl;
  }

  boost::archive::text_iarchive ia(infile);
  ia & *model;
  for (unsigned i = 0; i < model->morph_len; ++i) {
    Model *cnn_model = new Model();
    cnn_models->push_back(cnn_model);
  }

  model->InitParams(cnn_models);
  for (unsigned i = 0; i < model->morph_len; ++i) {
    ia & *(*cnn_models)[i];
  }

  cerr << "Loaded model from: " << filename << endl;
  infile.close();
}

