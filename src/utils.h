#ifndef UTILS_H_
#define UTILS_H_

#include <string>
#include <sstream>
#include <vector>

using namespace std;

string BOW = "<s>", EOW = "</s>";
unsigned MAX_PRED_LEN = 100;

vector<string> split_line(const string& line, char delim);

#endif
