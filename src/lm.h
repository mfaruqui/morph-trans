#ifndef LM_H_
#define LM_H_

#include "utils.h"

#include <boost/functional/hash.hpp>
#include <unordered_map>

using namespace std;

class LM {
 public:
  unordered_map<string, unsigned> char_to_id;
  unordered_map<unsigned, string> id_to_char;

  LM (string& lm_model_file, unordered_map<string, unsigned>& char_id,
      unordered_map<unsigned, string>& id_char);
  float LogProbSeq(vector<unsigned>& seq);
  float LogProb(vector<unsigned>& seq);
  size_t HashSeq(vector<unsigned>& seq);

 private:
  unordered_map<size_t, float> lp, b;
};

void
PrintSeq(vector<unsigned>& seq, unordered_map<unsigned, string>& id_to_char);

#endif
