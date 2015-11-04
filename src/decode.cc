#include "decode.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

string BOW = "<s>", EOW = "</s>";
int MAX_PRED_LEN = 100;

void Decode(const unsigned& morph_id,
            unordered_map<string, unsigned>& char_to_id,
            const vector<unsigned>& input_ids,
            vector<unsigned>* pred_target_ids, SepMorph* model) {
  ComputationGraph cg;
  model->AddParamsToCG(morph_id, &cg);

  Expression encoded_word_vec;
  model->RunFwdBwd(morph_id, input_ids, &encoded_word_vec, &cg);
  model->TransformEncodedInput(&encoded_word_vec);

  unsigned out_index = 1;
  unsigned pred_index = char_to_id[BOW];
  model->output_forward[morph_id].start_new_sequence();
  while (pred_target_ids->size() < MAX_PRED_LEN) {
    pred_target_ids->push_back(pred_index);
    if (pred_index == char_to_id[EOW]) {
      return;  // If the end is found, break from the loop and return
    }

    Expression prev_output_vec = lookup(cg, model->char_vecs[morph_id], pred_index);
    Expression input, input_char_vec;
    if (out_index < input_ids.size()) {
      input_char_vec = lookup(cg, model->char_vecs[morph_id], input_ids[out_index]);
    } else {
      input_char_vec = lookup(cg, model->eps_vecs[morph_id],
                              min(unsigned(out_index - input_ids.size()),
                                           model->max_eps - 1));
    }
    input = concatenate({encoded_word_vec, prev_output_vec, input_char_vec});
    Expression hidden = model->output_forward[morph_id].add_input(input);

    Expression out;
    model->ProjectToOutput(hidden, &out);
    vector<float> dist = as_vector(cg.incremental_forward());
    pred_index = distance(dist.begin(), max_element(dist.begin(),
                                                    dist.end()));;
    out_index++;
  }
}

void
EnsembleDecode(const unsigned& morph_id, unordered_map<string, unsigned>& char_to_id,
               const vector<unsigned>& input_ids, 
               vector<unsigned>* pred_target_ids, vector<SepMorph>* ensmb_model) {
  ComputationGraph cg;

  unsigned ensmb = ensmb_model->size();
  vector<Expression> encoded_word_vecs;
  for (unsigned i = 0; i < ensmb; ++i) {
    Expression encoded_word_vec;
    auto& model = (*ensmb_model)[i];
    model.AddParamsToCG(morph_id, &cg);
    model.RunFwdBwd(morph_id, input_ids, &encoded_word_vec, &cg);
    model.TransformEncodedInput(&encoded_word_vec);
    encoded_word_vecs.push_back(encoded_word_vec);
    model.output_forward[morph_id].start_new_sequence();
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
      auto& model = (*ensmb_model)[ensmb_id];
      Expression prev_output_vec = lookup(cg, model.char_vecs[morph_id], pred_index);
      Expression input, input_char_vec;
      if (out_index < input_ids.size()) {
        input_char_vec = lookup(cg, model.char_vecs[morph_id], input_ids[out_index]);
      } else {
        input_char_vec = lookup(cg, model.eps_vecs[morph_id],
                                min(unsigned(out_index - input_ids.size()),
                                             model.max_eps - 1));
      }
      input = concatenate({encoded_word_vecs[ensmb_id], prev_output_vec,
                           input_char_vec});

      Expression hidden = model.output_forward[morph_id].add_input(input);
      Expression out;
      model.ProjectToOutput(hidden, &out);
      ensmb_out.push_back(log_softmax(out));
    }

    Expression out = sum(ensmb_out) / ensmb_out.size();
    vector<float> dist = as_vector(cg.incremental_forward());
    pred_index = distance(dist.begin(), max_element(dist.begin(), dist.end()));
    out_index++;
  }
}
