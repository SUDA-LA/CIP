#pragma once
#include<iostream>
#include<string>
using namespace std;
#include<vector>
class dataset;
class sentence
{
public:
	friend dataset;
	vector<string> word;
	vector<string> tag;
	vector<vector<string>> word_char;
	~sentence() {}
};

