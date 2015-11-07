#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/training.h"
#include "cnn/gpu-ops.h"
#include "cnn/expr.h"

#include "lm.h"
#include "utils.h"
#include "lm-sep-morph.h"

#include <iostream>
#include <fstream>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  string vocab_filename = argv[1];  // vocabulary of words/characters
  string morph_filename = argv[2];
  string test_filename = argv[3];
  string lm_model_filename = argv[4];

  unordered_map<string, unsigned> char_to_id, morph_to_id;
  unordered_map<unsigned, string> id_to_char, id_to_morph;

  ReadVocab(vocab_filename, &char_to_id, &id_to_char);
  unsigned vocab_size = char_to_id.size();
  ReadVocab(morph_filename, &morph_to_id, &id_to_morph);
  unsigned morph_size = morph_to_id.size();

  vector<string> test_data;  // Read the dev file in a vector
  ReadData(test_filename, &test_data);

  LM lm(lm_model_filename, char_to_id);

  vector<vector<Model*> > ensmb_m;
  vector<LMSepMorph> ensmb_nn;
  for (unsigned i = 0; i < argc - 5; ++i) {
    vector<Model*> m;
    LMSepMorph nn;
    string f = argv[i + 5];
    Read(f, &nn, &m);
    ensmb_m.push_back(m);
    ensmb_nn.push_back(nn);
  }

  // Read the test file and output predictions for the words.
  string line;
  double correct = 0, total = 0;
  vector<LMSepMorph*> object_pointers;
  for (unsigned i = 0; i < ensmb_nn.size(); ++i) {
    object_pointers.push_back(&ensmb_nn[i]);
  }
  for (string& line : test_data) {
    vector<string> items = split_line(line, '|');
    vector<unsigned> input_ids, target_ids, pred_target_ids;
    input_ids.clear(); target_ids.clear(); pred_target_ids.clear();
    for (const string& ch : split_line(items[0], ' ')) {
      input_ids.push_back(char_to_id[ch]);
    }
    for (const string& ch : split_line(items[1], ' ')) {
      target_ids.push_back(char_to_id[ch]);
    }
    unsigned morph_id = morph_to_id[items[2]];
    EnsembleDecode(morph_id, char_to_id, input_ids, &pred_target_ids, &lm,
                   &object_pointers);

    string prediction = "";
    for (unsigned i = 0; i < pred_target_ids.size(); ++i) {
      prediction += id_to_char[pred_target_ids[i]];
      if (i != pred_target_ids.size() - 1) {
        prediction += " ";
      }
    }
    if (prediction == items[1]) {
      correct += 1;     
    } else {
      cout << "GOLD: " << line << endl;
      cout << "PRED: " << items[0] << "|" << prediction << "|" << items[2] << endl;
    }
    total += 1;
  }
  cerr << "Prediction Accuracy: " << correct / total << endl;
  return 1;
}
