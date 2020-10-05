#pragma once
#include<iostream>
#include<unordered_map>
#include<map>
#include<vector>
#include"dataset.h"
#include"windows.h"
using namespace std;
class linear_model
{
public:
	void create_feature_space();
	void online_training(bool averaged, bool shuffle,int iterator, int exitor);
	linear_model(string train_,string dev_,string test_);
	//存储。
	void save_file(int i);
	~linear_model();
private:
	//基础数据集。
	dataset train;
	dataset dev;
	dataset test;
	unordered_map<string, int> model;//特征空间。
	map<string, int> tag;//词性
	vector<int> w;
	vector<int> v;
	vector<int> update_time;
	vector<string> value;
	//创建特征空间。
	vector<string> create_feature(const sentence &sentence, int pos);
	//在线算法
	string maxscore_tag(const sentence &sen, int pos,bool);
	vector<int> get_id(vector<string> &f);
	void update_weight(const sentence &sen, int pos, const string &max_tag,const string &correct_tag,int correct_time);
	int count_score(int offset, vector<int> &fv,bool);
	//评价。
	double evaluate(dataset&,bool );
};

