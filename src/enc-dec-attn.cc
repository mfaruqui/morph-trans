#include "enc-dec-attn.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

string BOW = "<s>", EOW = "</s>";
int MAX_PRED_LEN = 100;
float NEG_INF = numeric_limits<int>::min();

EncDecAttn::EncDecAttn(const unsigned& char_length, const unsigned& hidden_length,
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

void EncDecAttn::InitParams(vector<Model*>* m) {
  for (unsigned i = 0; i < morph_len; ++i) {
    input_forward.push_back(LSTMBuilder(layers, char_len, hidden_len, (*m)[i]));
    input_backward.push_back(LSTMBuilder(layers, char_len, hidden_len, (*m)[i]));
    output_forward.push_back(LSTMBuilder(layers, char_len, hidden_len, (*m)[i]));

    phidden_to_output.push_back((*m)[i]->add_parameters({vocab_len, 2 * hidden_len}));
    phidden_to_output_bias.push_back((*m)[i]->add_parameters({vocab_len, 1}));

    char_vecs.push_back((*m)[i]->add_lookup_parameters(vocab_len, {char_len}));

    ptransform_encoded.push_back((*m)[i]->add_parameters({hidden_len,
                                                          2 * hidden_len}));
    ptransform_encoded_bias.push_back((*m)[i]->add_parameters({hidden_len, 1}));

    pcompress_hidden.push_back((*m)[i]->add_parameters({hidden_len,
                                                        2 * hidden_len}));
    pcompress_hidden_bias.push_back((*m)[i]->add_parameters({hidden_len, 1}));
    //eps_vecs.push_back((*m)[i]->add_lookup_parameters(max_eps, {char_len}));
  }
}

void EncDecAttn::AddParamsToCG(const unsigned& morph_id, ComputationGraph* cg) {
  input_forward[morph_id].new_graph(*cg);
  input_backward[morph_id].new_graph(*cg);
  output_forward[morph_id].new_graph(*cg);

  hidden_to_output = parameter(*cg, phidden_to_output[morph_id]);
  hidden_to_output_bias = parameter(*cg, phidden_to_output_bias[morph_id]);

  transform_encoded = parameter(*cg, ptransform_encoded[morph_id]);
  transform_encoded_bias = parameter(*cg, ptransform_encoded_bias[morph_id]);

  compress_hidden = parameter(*cg, pcompress_hidden[morph_id]);
  compress_hidden_bias = parameter(*cg, pcompress_hidden_bias[morph_id]);
}

void EncDecAttn::RunFwdBwd(const unsigned& morph_id,
                         const vector<unsigned>& inputs,
                         Expression* hidden, vector<Expression>* all_hidden,
                         ComputationGraph *cg) {
  vector<Expression> input_vecs;
  for (const unsigned& input_id : inputs) {
    input_vecs.push_back(lookup(*cg, char_vecs[morph_id], input_id));
  }

  // Run forward LSTM
  Expression forward_unit;
  vector<Expression> fwd_units;
  input_forward[morph_id].start_new_sequence();
  for (unsigned i = 0; i < input_vecs.size(); ++i) {
    forward_unit = input_forward[morph_id].add_input(input_vecs[i]);
    fwd_units.push_back(forward_unit);
  }

  // Run backward LSTM
  Expression backward_unit;
  vector<Expression> bwd_units;
  input_backward[morph_id].start_new_sequence();
  for (int i = input_vecs.size() - 1; i >= 0; --i) {
    backward_unit = input_backward[morph_id].add_input(input_vecs[i]);
    bwd_units.push_back(backward_unit);
  }

  // Concatenate the forward and back hidden layers
  *hidden = concatenate({forward_unit, backward_unit});
  for (unsigned i = 0; i < fwd_units.size(); ++i) {
    all_hidden->push_back(affine_transform({compress_hidden_bias, compress_hidden,
                                            concatenate({fwd_units[i], bwd_units[i]})}));
  }
}

Expression EncDecAttn::GetAvgAttnLayer(const Expression& hidden,
                           const vector<Expression>& all_input_hidden) const {
  vector<Expression> prob;
  for (Expression ih: all_input_hidden) {
    prob.push_back(dot_product(hidden, ih));
  }
  //cerr << "here";
  Expression p = softmax(concatenate(prob));
  Expression hidden_mat = concatenate_cols(all_input_hidden);
  return hidden_mat * p;
}

void EncDecAttn::TransformEncodedInput(Expression* encoded_input) const {
  *encoded_input = affine_transform({transform_encoded_bias,
                                     transform_encoded, *encoded_input});
}

void EncDecAttn::ProjectToOutput(const Expression& hidden,
                                 const vector<Expression>& all_input_hidden,
                                 Expression* out) const {
  Expression avg_attn = GetAvgAttnLayer(hidden, all_input_hidden);
  //cerr << "not here";
  *out = affine_transform({hidden_to_output_bias, hidden_to_output,
                           concatenate({hidden, avg_attn})});
}

Expression EncDecAttn::ComputeLoss(const vector<Expression>& hidden_units,
                                   const vector<unsigned>& targets,
                                   const vector<Expression>& all_input_hidden) const {
  assert(hidden_units.size() == targets.size());
  vector<Expression> losses;
  for (unsigned i = 0; i < hidden_units.size(); ++i) {
    Expression out;
    ProjectToOutput(hidden_units[i], all_input_hidden, &out);
    losses.push_back(pickneglogsoftmax(out, targets[i]));
  }
  return sum(losses);
}

float EncDecAttn::Train(const unsigned& morph_id, const vector<unsigned>& inputs,
                      const vector<unsigned>& outputs, AdadeltaTrainer* ada_gd) {
  ComputationGraph cg;
  AddParamsToCG(morph_id, &cg);

  // Encode and Transform to feed into decoder
  Expression encoded_input_vec;
  vector<Expression> all_input_hidden;
  RunFwdBwd(morph_id, inputs, &encoded_input_vec, &all_input_hidden, &cg);
  TransformEncodedInput(&encoded_input_vec);

  // Use this encoded word vector to predict the transformed word
  vector<Expression> input_vecs_for_dec;
  vector<unsigned> output_ids_for_pred;
  for (unsigned i = 0; i < outputs.size(); ++i) {
    if (i < outputs.size() - 1) { 
      // '</s>' will not be fed as input -- it needs to be predicted.
        input_vecs_for_dec.push_back(lookup(cg, char_vecs[morph_id], outputs[i]));
    }
    if (i > 0) {  // '<s>' will not be predicted in the output -- its fed in.
      output_ids_for_pred.push_back(outputs[i]);
    }
  }

  vector<Expression> decoder_hidden_units;
  vector<Expression> init;
  init.push_back(tanh(encoded_input_vec));
  init.push_back(encoded_input_vec);
  output_forward[morph_id].start_new_sequence(init);
  for (const auto& vec : input_vecs_for_dec) {
    decoder_hidden_units.push_back(output_forward[morph_id].add_input(vec));
  }
  Expression loss = ComputeLoss(decoder_hidden_units, output_ids_for_pred, all_input_hidden);

  float return_loss = as_scalar(cg.forward());
  cg.backward();
  ada_gd->update(1.0f);
  return return_loss;
}

void
EnsembleDecode(const unsigned& morph_id, unordered_map<string, unsigned>& char_to_id,
               const vector<unsigned>& input_ids,
               vector<unsigned>* pred_target_ids, vector<EncDecAttn*>* ensmb_model) {
  ComputationGraph cg;

  unsigned ensmb = ensmb_model->size();
  //vector<Expression> encoded_word_vecs;
  vector<vector<Expression> > all_input_hidden;
  for (unsigned i = 0; i < ensmb; ++i) {
    vector<Expression> input_hidden;
    Expression encoded_word_vec;
    auto model = (*ensmb_model)[i];
    model->AddParamsToCG(morph_id, &cg);
    model->RunFwdBwd(morph_id, input_ids, &encoded_word_vec, &input_hidden, &cg);
    model->TransformEncodedInput(&encoded_word_vec);
    all_input_hidden.push_back(input_hidden);
    //encoded_word_vecs.push_back(encoded_word_vec);
    
    vector<Expression> init;
    init.push_back(tanh(encoded_word_vec));
    init.push_back(encoded_word_vec);
    model->output_forward[morph_id].start_new_sequence(init);
  }

  unsigned out_index = 1;
  unsigned pred_index = char_to_id[BOW];
  while (pred_target_ids->size() < MAX_PRED_LEN) {
    vector<Expression> ensmb_out;
    pred_target_ids->push_back(pred_index);
    if (pred_index == char_to_id[EOW]) {
      return;  // If the end is found, break from the loop and return
    }

    for (unsigned ensmb_id = 0; ensmb_id < ensmb; ++ensmb_id) {
      auto model = (*ensmb_model)[ensmb_id];
      Expression prev_output_vec = lookup(cg, model->char_vecs[morph_id], pred_index);
      Expression input, input_char_vec;
      /*if (out_index < input_ids.size()) {
        input_char_vec = lookup(cg, model->char_vecs[morph_id], input_ids[out_index]);
      } else {
        input_char_vec = lookup(cg, model->eps_vecs[morph_id],
                                min(unsigned(out_index - input_ids.size()),
                                             model->max_eps - 1));
      }
      input = concatenate({encoded_word_vecs[ensmb_id], prev_output_vec,
                           input_char_vec});*/
      input = prev_output_vec;

      Expression hidden = model->output_forward[morph_id].add_input(input);
      Expression out;
      model->ProjectToOutput(hidden, all_input_hidden[ensmb_id], &out);
      ensmb_out.push_back(log_softmax(out));
    }

    Expression out = sum(ensmb_out) / ensmb_out.size();
    vector<float> dist = as_vector(cg.incremental_forward());
    pred_index = distance(dist.begin(), max_element(dist.begin(), dist.end()));
    out_index++;
  }
}

void
EnsembleBeamDecode(const unsigned& morph_id, const unsigned& beam_size, 
                   unordered_map<string, unsigned>& char_to_id,
                   const vector<unsigned>& input_ids,
                   vector<vector<unsigned> >* sequences, vector<float>* tm_scores,
                   vector<EncDecAttn*>* ensmb_model) {
  unsigned out_index = 1;
  unsigned ensmb = ensmb_model->size();
  ComputationGraph cg;

  // Compute stuff for every model in the ensemble.
  //vector<Expression> encoded_word_vecs;
  vector<vector<Expression> > all_input_hidden;
  vector<Expression> ensmb_out;
  for (unsigned ensmb_id = 0; ensmb_id < ensmb; ++ensmb_id) {
    auto& model = *(*ensmb_model)[ensmb_id];
    model.AddParamsToCG(morph_id, &cg);

    Expression encoded_word_vec;
    vector<Expression> input_hidden;
    model.RunFwdBwd(morph_id, input_ids, &encoded_word_vec, &input_hidden, &cg);
    model.TransformEncodedInput(&encoded_word_vec);
    //encoded_word_vecs.push_back(encoded_word_vec);
    all_input_hidden.push_back(input_hidden);

    vector<Expression> init;
    init.push_back(tanh(encoded_word_vec));
    init.push_back(encoded_word_vec);
    model.output_forward[morph_id].start_new_sequence(init);

    Expression prev_output_vec = lookup(cg, model.char_vecs[morph_id],
                                        char_to_id[BOW]);
    /*Expression input = concatenate({encoded_word_vecs[ensmb_id], prev_output_vec,
                                    lookup(cg, model.char_vecs[morph_id],
                                    input_ids[out_index])});*/
    Expression input = prev_output_vec;
    Expression hidden = model.output_forward[morph_id].add_input(input);
    Expression out;
    model.ProjectToOutput(hidden, input_hidden, &out);
    out = log_softmax(out);
    ensmb_out.push_back(out);
  }

  // Compute the average of the ensemble output.
  Expression out_dist = average(ensmb_out);
  vector<float> log_dist = as_vector(cg.incremental_forward());
  priority_queue<pair<float, unsigned> > init_queue;
  for (unsigned i = 0; i < log_dist.size(); ++i) {
    init_queue.push(make_pair(log_dist[i], i));
  }
  unsigned vocab_size = log_dist.size();

  // Initialise the beam_size sequences, scores, hidden states.
  vector<float> log_scores;
  vector<vector<RNNPointer> > prev_states;
  for (unsigned beam_id = 0; beam_id < beam_size; ++beam_id) {
    vector<unsigned> seq;
    seq.push_back(char_to_id[BOW]);
    seq.push_back(init_queue.top().second);
    sequences->push_back(seq);
    log_scores.push_back(init_queue.top().first);
   
    vector<RNNPointer> ensmb_states;
    for (unsigned ensmb_id = 0; ensmb_id < ensmb; ++ensmb_id) {
      auto& model = *(*ensmb_model)[ensmb_id];
      ensmb_states.push_back(model.output_forward[morph_id].state());
    }
    prev_states.push_back(ensmb_states);
    init_queue.pop();
  }

  vector<cnn::real> neg_inf(vocab_size, NEG_INF);
  Expression neg_inf_vec = cnn::expr::input(cg, {vocab_size}, &neg_inf);
 
  vector<bool> active_beams(beam_size, true);
  while (true) {
    out_index++;      
    priority_queue<pair<float, pair<unsigned, unsigned> > > probs_queue;
    vector<vector<RNNPointer> > curr_states;
    vector<Expression> out_dist;
    for (unsigned beam_id = 0; beam_id < beam_size; ++beam_id) {
      if (active_beams[beam_id]) {
        unsigned prev_out_char = (*sequences)[beam_id].back();
        vector<Expression> ensmb_out;
        vector<RNNPointer> ensmb_states;
        for (unsigned ensmb_id = 0; ensmb_id < ensmb; ensmb_id++) {
          auto& model = *(*ensmb_model)[ensmb_id];
          /*Expression input_char_vec;
          /if (out_index < input_ids.size()) { 
            input_char_vec = lookup(cg, model.char_vecs[morph_id], input_ids[out_index]);
          } else { 
            input_char_vec = lookup(cg, model.eps_vecs[morph_id],
                                    min(unsigned(out_index - input_ids.size()),
                                                 model.max_eps - 1));
          }*/

          Expression prev_out_vec = lookup(cg, model.char_vecs[morph_id], prev_out_char);
          //Expression input = concatenate({encoded_word_vecs[ensmb_id], prev_out_vec,
          //                                input_char_vec});
         
          Expression input = prev_out_vec;
          Expression hidden = model.output_forward[morph_id].add_input(
                                prev_states[beam_id][ensmb_id], input);
          ensmb_states.push_back(model.output_forward[morph_id].state());
          
          Expression out;
          model.ProjectToOutput(hidden, all_input_hidden[ensmb_id], &out);
          out = log_softmax(out);
          ensmb_out.push_back(out);
        }
        curr_states.push_back(ensmb_states);
        out_dist.push_back(average(ensmb_out));
      } else {
        vector<RNNPointer> dummy(ensmb, RNNPointer(0));
        curr_states.push_back(dummy);
        out_dist.push_back(neg_inf_vec);
      }
    }

    Expression all_scores = concatenate(out_dist);
    vector<float> log_dist = as_vector(cg.incremental_forward());

    for (unsigned index = 0; index < log_dist.size(); ++index) {
      unsigned beam_id = index / vocab_size;
      unsigned char_id = index % vocab_size;
      if (active_beams[beam_id]) {
        pair<unsigned, unsigned> location = make_pair(beam_id, char_id);
        probs_queue.push(pair<float, pair<unsigned, unsigned> >(
                         log_scores[beam_id] + log_dist[index], location));
      }
    }
      
    // Find the beam_size best now and update the variables.
    unordered_map<unsigned, vector<unsigned> > new_seq;
    for (unsigned beam_id = 0; beam_id < beam_size; ++beam_id) {
      if (active_beams[beam_id]) {
        float log_prob = probs_queue.top().first;
        pair<unsigned, unsigned> location = probs_queue.top().second;
        unsigned old_beam_id = location.first, char_id = location.second;

        vector<unsigned> seq = (*sequences)[old_beam_id];
        seq.push_back(char_id);
        new_seq[beam_id] = seq;
        log_scores[beam_id] = log_prob;  // Update the score

        prev_states[beam_id] = curr_states[old_beam_id];  // Update hidden state
        probs_queue.pop();
      }
    }
      
    // Update the sequences now.
    for (auto& it : new_seq) {
      (*sequences)[it.first] = it.second;
    }

    // Check if a sequence should be made inactive.
    for (unsigned beam_id = 0; beam_id < beam_size; ++beam_id) {
      if (active_beams[beam_id] && 
          ((*sequences)[beam_id].back() == char_to_id[EOW] ||
           (*sequences)[beam_id].size() > MAX_PRED_LEN)) {
        active_beams[beam_id] = false;
      }
    }

    // Check if all sequences are inactive.
    bool all_inactive = true;
    for (unsigned beam_id = 0; beam_id < beam_size; ++beam_id) {
      if (active_beams[beam_id]) {
        all_inactive = false;
        break;
      }
    }

    if (all_inactive) {
      *tm_scores = log_scores;
      return;
    }
  }
}

void Serialize(string& filename, EncDecAttn& model, vector<Model*>* cnn_models) {
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

void Read(string& filename, EncDecAttn* model, vector<Model*>* cnn_models) {
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

