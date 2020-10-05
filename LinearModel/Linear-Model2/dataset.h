#pragma once
#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include"sentence.h"
using namespace std;
class dataset
{
public:
	void read_data(const string &file_name);
	vector<sentence> sentences;
	string name;
	int sentence_count = 0, word_count = 0;
	void shuffle();
	
	dataset();
	~dataset();
};

	