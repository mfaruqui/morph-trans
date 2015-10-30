#include "utils.h"

vector<string> split_line(const string& line, char delim) {
  vector<string> words;
  stringstream ss(line);
  string item;
  while (std::getline(ss, item, delim)) {
    if (!item.empty())
      words.push_back(item);
  }
  return words;
}

void ReadVocab(string& filename, unordered_map<string, unsigned>* item_to_id,
               unordered_map<unsigned, string>* id_to_item) {
  ifstream vocab_file(filename);
  vector<string> chars;
  if (vocab_file.is_open()) {  // Reading the vocab file
    string line;
    getline(vocab_file, line);
    chars = split_line(line, ' ');
  } else {
    cerr << "File opening failed" << endl;
  }
  unsigned char_id = 0;
  for (const string& ch : chars) {
    (*item_to_id)[ch] = char_id;
    (*id_to_item)[char_id] = ch;
    char_id++;
  }
  vocab_file.close();
}

void ReadData(string& filename, vector<string>* data) {
  // Read the training file in a vector
  ifstream train_file(filename);
  if (train_file.is_open()) {
    string line;
    while (getline(train_file, line)) {
      data->push_back(line);
    }
  }
  train_file.close();
}
