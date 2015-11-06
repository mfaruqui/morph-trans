#include "lm.h"

LM::LM(string& lm_model_file, unordered_map<string, unsigned>& char_id) {
  ifstream model_file(lm_model_file);
  if (model_file.is_open()) {
    string line;
    char_to_id = char_id;
    while(getline(model_file, line)) {
      vector<string> items = split_line(line, '\t');
      if (items.size() == 2) {
        vector<unsigned> seq;
        for (string& ch : split_line(items[1], ' ')) {
          seq.push_back(char_to_id[ch]);
        }
        lp[HashSeq(seq)] = atof(items[0].c_str());
      } else if (items.size() == 3) {
        vector<unsigned> seq;
        for (string& ch : split_line(items[1], ' ')) {
          seq.push_back(char_to_id[ch]);
        }
        lp[HashSeq(seq)] = atof(items[0].c_str());
        b[HashSeq(seq)] = atof(items[2].c_str());
      }
    }
    model_file.close();
    cerr << "Language model loaded from: " << lm_model_file << endl;
    cerr << "LM size: " << lp.size() << endl;
  } else {
    cerr << "File opening failed" << endl;
  }
}

size_t LM::HashSeq(vector<unsigned>& seq) {
  return boost::hash_range(seq.begin(), seq.end());
}

/* LogProbSeq(w1, w2, ..., wn) = LogProbSeq(w2, w2, ..., wn)
                                + backoff(w1, w2, ..., wn-1);
   http://cmusphinx.sourceforge.net/wiki/sphinx4:standardgrammarformats

   Assumes that unigrams are always present in lp[].
*/
float LM::LogProbSeq(vector<unsigned>& seq) {
  size_t hash = HashSeq(seq);
  auto it_seq = lp.find(hash);
  if (it_seq == lp.end()) {
    vector<unsigned> backoff_seq(seq.begin(), seq.end() - 1);
    float backoff;
    auto it_backoff = b.find(hash);
    if (it_backoff == b.end()) {
      backoff = 0.0;
    } else {
      backoff = it_backoff->second;
    }
    vector<unsigned> small(seq.begin() + 1, seq.end());

    float log_prob = backoff + LogProbSeq(small);
    lp[hash] = log_prob;
    return log_prob;
  } else {
    if (it_seq->second != it_seq->second) {
      cerr << endl << "Bad LM prob:" << it_seq->second << endl;
    }
    return it_seq->second;  // Return the log prob 
  }
}
