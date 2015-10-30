#ifndef UTILS_H_
#define UTILS_H_

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <unordered_map>

using namespace std;

vector<string> split_line(const string& line, char delim);

void ReadVocab(string& filename, unordered_map<string, unsigned>* item_to_id,
               unordered_map<unsigned, string>* id_to_item);

void ReadData(string& filename, vector<string>* data);

#endif
