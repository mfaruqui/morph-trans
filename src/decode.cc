#include "decode.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

string BOW = "<s>", EOW = "</s>";
int MAX_PRED_LEN = 100;

template<typename T>
void Decode(unordered_map<string, unsigned>& char_to_id,
            const vector<unsigned>& input_ids,
            vector<unsigned>* pred_target_ids, T* model) {
  ComputationGraph cg;
  model->AddParamsToCG(&cg);

  Expression encoded_word_vec;
  model->RunFwdBwd(input_ids, &encoded_word_vec, &cg);
  model->TransformEncodedInputForDecoding(&encoded_word_vec);

  unsigned out_index = 1;
  unsigned pred_index = char_to_id[BOW];
  model->output_forward.start_new_sequence();
  while (pred_target_ids->size() < MAX_PRED_LEN) {
    Expression prev_output_vec = lookup(cg, model->char_vecs, pred_index);
    pred_target_ids->push_back(pred_index);
    Expression input, input_char_vec;
    if (out_index < input_ids.size()) {
      input_char_vec = lookup(cg, model->char_vecs, input_ids[out_index]);
    } else {
      input_char_vec = model->EPS;
    }
    input = concatenate({encoded_word_vec, prev_output_vec, input_char_vec});
    Expression hidden = model->output_forward.add_input(input);

    Expression out;
    model->ProjectToOutput(hidden, &out);
    vector<float> dist = as_vector(cg.incremental_forward());
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
    out_index++;
  }
}

template<typename T> void
EnsembleDecode(unordered_map<string, unsigned>& char_to_id,
               const vector<unsigned>& input_ids, 
               vector<unsigned>* pred_target_ids, Ensemble<T>* ensmb_model) {
  ComputationGraph cg;

  // WHY IS THIS NOT ABLE TO FIND THIS METHOD HERE
  //ensmb_model->AddParamsToCG(&cg);

  unsigned ensmb = ensmb_model->ensmb;
  vector<Expression> encoded_word_vecs;
  for (unsigned i = 0; i < ensmb; ++i) {
    Expression encoded_word_vec;
    auto& model = ensmb_model->models[i];
    model.AddParamsToCG(&cg);
    model.RunFwdBwd(input_ids, &encoded_word_vec, &cg);
    model.TransformEncodedInputForDecoding(&encoded_word_vec);
    encoded_word_vecs.push_back(encoded_word_vec);
    model.output_forward.start_new_sequence();
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
      auto& model = ensmb_model->models[ensmb_id];
      Expression prev_output_vec = lookup(cg, model.char_vecs, pred_index);
      Expression input, input_char_vec;
      if (out_index < input_ids.size()) {
        input_char_vec = lookup(cg, model.char_vecs, input_ids[out_index]);
      } else {
        input_char_vec = model.EPS;
      }
      input = concatenate({encoded_word_vecs[ensmb_id], prev_output_vec,
                           input_char_vec});

      Expression hidden = model.output_forward.add_input(input);
      Expression out;
      model.ProjectToOutput(hidden, &out);
      ensmb_out.push_back(softmax(out));
    }

    Expression out = sum(ensmb_out) / ensmb_out.size();
    vector<float> dist = as_vector(cg.incremental_forward());
    pred_index = distance(dist.begin(), max_element(dist.begin(), dist.end()));
    out_index++;
  }
}

template void
Decode<MorphTrans>(unordered_map<string, unsigned>& char_to_id,
                   const vector<unsigned>& input_ids,
                   vector<unsigned>* pred_target_ids, MorphTrans* model);

template void
EnsembleDecode<MorphTrans>(unordered_map<string, unsigned>& char_to_id,
                           const vector<unsigned>& input_ids,
                           vector<unsigned>* pred_target_ids,
                           Ensemble<MorphTrans>* ensmb_model);
