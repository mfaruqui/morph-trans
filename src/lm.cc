#include "lm.h"

LM::LM(string& lm_model_file, unordered_map<string, unsigned>& char_id,
       unordered_map<unsigned, string>& id_char) {
  ifstream model_file(lm_model_file);
  if (model_file.is_open()) {
    string line;
    char_to_id = char_id;
    id_to_char = id_char;
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
    cerr << "LM loaded from: " << lm_model_file << endl;
    cerr << "LM size: " << lp.size() << endl;
  } else {
    cerr << "File opening failed" << endl;
    exit(0);
  }
}

size_t LM::HashSeq(vector<unsigned>& seq) {
  return boost::hash_range(seq.begin(), seq.end());
}

void
PrintSeq(vector<unsigned>& seq, unordered_map<unsigned, string>& id_to_char) {
  for (unsigned i = 0; i < seq.size(); ++i) {
    cerr << id_to_char[seq[i]] << " ";
  }
  cerr << endl;
}

/* LogProbSeq(w1, w2, ..., wn) = LogProbSeq(w2, w3, ..., wn)
                                + backoff(w1, w2, ..., wn-1);
   http://cmusphinx.sourceforge.net/wiki/sphinx4:standardgrammarformats

   Assumes that unigrams are always present in lp[].
*/
float LM::LogProb(vector<unsigned>& seq) {
  size_t seq_hash = HashSeq(seq);
  auto it_seq = lp.find(seq_hash);
  if (it_seq == lp.end()) {
    vector<unsigned> backoff_seq(seq.begin(), seq.end() - 1);
    size_t backoff_hash = HashSeq(backoff_seq);
    auto it_backoff = b.find(backoff_hash);

    float backoff;
    if (it_backoff == b.end()) {
      backoff = 0.0;
    } else {
      backoff = it_backoff->second;
    }

    vector<unsigned> small(seq.begin() + 1, seq.end());
    lp[seq_hash] = backoff + LogProb(small);
    return lp[seq_hash];
  } else {
    return it_seq->second;  // Return the log prob 
  }
}

float LM::LogProbSeq(vector<unsigned>& seq) {
  float score = 0.;
  for (unsigned i = 1; i <= seq.size(); ++i) {
    vector<unsigned> temp(seq.begin(), seq.begin()+ i);
    score += LogProb(temp);
  }
  return score;
}
