/*
This file outputs all the strings in the beam.
*/
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/training.h"
#include "cnn/gpu-ops.h"
#include "cnn/expr.h"

#include "utils.h"
#include "sep-morph.h"

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
  string lm_filename = argv[4];
  unsigned beam_size = atoi(argv[5]);

  unordered_map<string, unsigned> char_to_id, morph_to_id;
  unordered_map<unsigned, string> id_to_char, id_to_morph;

  ReadVocab(vocab_filename, &char_to_id, &id_to_char);
  unsigned vocab_size = char_to_id.size();
  ReadVocab(morph_filename, &morph_to_id, &id_to_morph);
  unsigned morph_size = morph_to_id.size();

  vector<string> test_data;  // Read the dev file in a vector
  ReadData(test_filename, &test_data);

  vector<vector<Model*> > ensmb_m;
  vector<SepMorph> ensmb_nn;
  for (unsigned i = 0; i < argc - 6; ++i) {
    vector<Model*> m;
    SepMorph nn;
    string f = argv[i + 6];
    Read(f, &nn, &m);
    ensmb_m.push_back(m);
    ensmb_nn.push_back(nn);
  }

  // Read the test file and output predictions for the words.
  string line;
  double correct = 0, total = 0;
  vector<SepMorph*> object_pointers;
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

    vector<vector<unsigned> > pred_beams;
    vector<float> beam_score;
    unsigned morph_id = morph_to_id[items[2]];
    EnsembleBeamDecode(morph_id, beam_size, char_to_id, input_ids, &pred_beams,
                       &beam_score, &object_pointers);

    cout << "GOLD: " << line << endl;
    for (unsigned beam_id = 0; beam_id < beam_size; ++beam_id) {
      pred_target_ids = pred_beams[beam_id];
      string prediction = "";
      for (unsigned i = 0; i < pred_target_ids.size(); ++i) { 
        prediction += id_to_char[pred_target_ids[i]];
        if (i != pred_target_ids.size() - 1) {
          prediction += " ";
        }
      }
      cout << "PRED: " << prediction << " " << beam_score[beam_id] << endl;
    }
  }
  return 1;
}
