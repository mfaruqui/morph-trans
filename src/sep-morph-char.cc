#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/training.h"
#include "cnn/gpu-ops.h"
#include "cnn/expr.h"

#include "utils.h"
#include "decode.h"
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

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  string vocab_filename = argv[1];  // vocabulary of words/characters
  string morph_filename = argv[2];
  string train_filename = argv[3];
  string test_filename = argv[4];
  unsigned char_size = atoi(argv[5]);
  unsigned hidden_size = atoi(argv[6]);
  unsigned num_iter = atoi(argv[7]);
  float reg_strength = atof(argv[8]);
  unsigned layers = atoi(argv[9]);

  ifstream vocab_file(vocab_filename);
  vector<string> chars;
  if (vocab_file.is_open()) {  // Reading the vocab file
    string line;
    getline(vocab_file, line);
    chars = split_line(line, ' ');
  } else {
    cerr << "File opening failed" << endl;
  }
  unordered_map<string, unsigned> char_to_id;
  unordered_map<unsigned, string> id_to_char;
  unsigned char_id = 0;
  for (const string& ch : chars) {
    char_to_id[ch] = char_id;
    id_to_char[char_id] = ch;
    char_id++;
  }
  unsigned vocab_size = char_to_id.size();

  ifstream morph_file(morph_filename);
  vector<string> morph_attrs;
  if (morph_file.is_open()) {  // Reading the vocab file
    string line;
    getline(morph_file, line);
    morph_attrs = split_line(line, ' ');
  } else {
    cerr << "File opening failed" << endl;
  }
  unordered_map<string, unsigned> morph_to_id;
  unordered_map<unsigned, string> id_to_morph;
  unsigned morph_id = 0;
  for (const string& ch : morph_attrs) {
    morph_to_id[ch] = morph_id;
    id_to_morph[morph_id] = ch;
    morph_id++;
  }
  unsigned morph_size = morph_to_id.size();

  vector<Model*> m;
  vector<AdadeltaTrainer> optimizer;
  vector<MorphTrans> nn;
  for (unsigned i = 0; i < morph_size; ++i) {
    m.push_back(new Model());
    AdadeltaTrainer ada(m[i], reg_strength);
    optimizer.push_back(ada);
    MorphTrans neural(char_size, hidden_size, vocab_size, layers, m[i]);
    nn.push_back(neural);
  }

  // Read the training file in a vector
  vector<string> train_data;
  ifstream train_file(train_filename);
  if (train_file.is_open()) {
    string line;
    while (getline(train_file, line)) {
      train_data.push_back(line);
    }
  }
  train_file.close();

  // Read the test file in a vector
  vector<string> test_data;
  ifstream test_file(test_filename);
  if (test_file.is_open()) {
    string line;
    while (getline(test_file, line)) {
      test_data.push_back(line);
    }
  }
  test_file.close();

  // Read the training file and train the model
  for (unsigned iter = 0; iter < num_iter; ++iter) {
    unsigned line_id = 0;
    random_shuffle(train_data.begin(), train_data.end());
    vector<float> loss(morph_size, 0.0f);
    for (string& line : train_data) {
      vector<string> items = split_line(line, '|');
      vector<unsigned> input_ids, target_ids;
      input_ids.clear(); target_ids.clear();
      for (const string& ch : split_line(items[0], ' ')) {
        input_ids.push_back(char_to_id[ch]);
      }
      for (const string& ch : split_line(items[1], ' ')) {
        target_ids.push_back(char_to_id[ch]);
      }
      unsigned morph_id = morph_to_id[items[2]];
      loss[morph_id] += nn[morph_id].Train(input_ids, target_ids, &optimizer[morph_id]);
      cerr << ++line_id << "\r";
    }

    cerr << "Iter " << iter + 1 << " ";
    for (unsigned i = 0; i < loss.size(); ++i) {
      cerr << loss[i] << " ";
    }
    cerr << "Sum: " << accumulate(loss.begin(), loss.end(), 0.) << endl;

    // Read the test file and output predictions for the words.
    string line;
    double correct = 0, total = 0;
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
      nn[morph_id].Predict(input_ids, char_to_id, &pred_target_ids);

      string prediction = "";
      for (unsigned i = 0; i < pred_target_ids.size(); ++i) {
        prediction += id_to_char[pred_target_ids[i]];
        if (i != pred_target_ids.size() - 1) {
          prediction += " ";
        }
      }
      if (prediction == items[1]) {
        correct += 1;
      } else {  // If wrong, print prediction and correct answer
        if (iter == num_iter - 1) {
          cout << items[0] << '|' << items[1] << '|' << items[2] << endl;
          cout << items[0] << '|' << prediction << '|' << items[2] << endl;
        }
      }
      total += 1;
    }
    cerr << "Prediction Accuracy: " << correct / total << endl;
  }
  return 1;
}
