#include "decode.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

string BOW = "<s>", EOW = "</s>";
int MAX_PRED_LEN = 100;

void Decode(const Expression& encoded_word_vec,
            unordered_map<string, unsigned>& char_to_id,
            const vector<unsigned>& input_ids,
            ProjToOutput& proj_to_vocab, LSTMBuilder* decoder,
            LookupParameters* char_vecs, Expression& EPS,
            vector<unsigned>* pred_target_ids, ComputationGraph* cg) {
  Expression input_word_vec = lookup(*cg, char_vecs, char_to_id[BOW]);
  pred_target_ids->push_back(char_to_id[BOW]);
  decoder->start_new_sequence();
  unsigned out_index = 1;
  while (pred_target_ids->size() < MAX_PRED_LEN) {
    Expression input;
    if (out_index < input_ids.size()) {
      input = concatenate({encoded_word_vec, input_word_vec,
                          lookup(*cg, char_vecs, input_ids[out_index])});
    } else {
      input = concatenate({encoded_word_vec, input_word_vec, EPS});
    }
    Expression hidden = decoder->add_input(input);
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
