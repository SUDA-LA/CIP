#include "linear_model.h"
#include<numeric>
//实例化特征
vector<string> linear_model::create_feature(const sentence &sentence, int pos)
{
	string word = sentence.word[pos];//当前词。
	string word_char_first = sentence.word_char[pos][0];//当前词的第一个元素。
	string word_char_last = sentence.word_char[pos][sentence.word_char[pos].size() - 1];//当前词的最后一个元素。
	string word_m1;
	string word_char_m1m1;
	string word_p1;
	string word_char_p1_first;
	int word_count = sentence.word.size();//当前句的总词数。
	if (pos == 0)
	{
		word_m1 = "$$";
		word_char_m1m1 = "$";
	}
	else
	{
		word_m1 = sentence.word[pos - 1];
		word_char_m1m1 = sentence.word_char[pos - 1][(sentence.word_char[pos - 1].size() - 1)];
	}
	if (pos == word_count - 1)
	{
		word_p1 = "##";
		word_char_p1_first = "#";
	}
	else
	{
		word_p1 = sentence.word[pos + 1];
		word_char_p1_first = sentence.word_char[pos + 1][0];
	}
	vector<string> f;
	f.reserve(50);
	f.emplace_back("02:*" + word);
	f.emplace_back("03:*" + word_m1);
	f.emplace_back("04:*" + word_p1);
	f.emplace_back("05:*" + word + "*" + word_char_m1m1);
	f.emplace_back("06:*" + word + "*" + word_char_p1_first);
	f.emplace_back("07:*" + word_char_first);
	f.emplace_back("08:*" + word_char_last);
	int pos_word_len = sentence.word_char[pos].size();
	for (int k = 1; k < pos_word_len - 1; k++)
	{
		string cik = sentence.word_char[pos][k];
		f.emplace_back("09:*" + cik);
		f.emplace_back("10:*" + word_char_first + "*" + cik);
		f.emplace_back("11:*" + word_char_last + "*" + cik);
		string cikp1 = sentence.word_char[pos][k + 1];
		if (cik == cikp1)
		{
			f.emplace_back("13:*" + cik + "*" + "consecutive");
		}
	}
	if (sentence.word_char[pos].size() > 1)
	{
		if (sentence.word_char[pos][0] == sentence.word_char[pos][1])
			f.emplace_back("13:*" + sentence.word_char[pos][0] + "*" + "consecutive");
	}
	if (pos_word_len == 1)
	{
		f.emplace_back("12:*" + word + "*" + word_char_m1m1 + "*" + word_char_p1_first);
	}
	for (int k = 0; k <pos_word_len; k++)
	{
		if (k >=4)break;
		f.emplace_back("14:*" + accumulate(sentence.word_char[pos].begin(), sentence.word_char[pos].begin() + k + 1, string("")));
		f.emplace_back("15:*" + accumulate(sentence.word_char[pos].end() - (k + 1), sentence.word_char[pos].end(), string("")));
	}
	return f;
}
//创建特征空间
void linear_model::create_feature_space()
{
	int count_feature = 0, count_tag = 0;
	for (auto z = train.sentences.begin(); z != train.sentences.end(); ++z)
	{
		for (int i = 0; i < z->word.size(); i++)
		{
			vector <string> f=create_feature(*z, i);
			for (auto q = f.begin(); q != f.end(); q++)
			{
				if (model.find(*q) == model.end())//如果不在词性里面。
				{
					model.insert({ *q, count_feature });
					value.emplace_back(*q);
					count_feature++;
				}
			}
			if (tag.find(z->tag[i]) == tag.end())
			{
				tag.insert({ z->tag[i] , count_tag });
				count_tag++;
			}
		}
	}
	w.reserve(tag.size()*model.size());
	v.reserve(tag.size()*model.size());
	update_time.reserve(tag.size()*model.size());
	for (int i = 0; i < tag.size()*model.size(); i++)
	{
		w.emplace_back(0);
		v.emplace_back(0);
		update_time.emplace_back(0);
	}
	cout << "the total number of features is " << model.size() << endl;
	cout << "the total number of tags is " << tag.size() << endl;

}
vector<int> linear_model::get_id(vector<string> &f)
{
	vector<int> fv;
	for (auto q = f.begin(); q != f.end(); q++)
	{
		auto t = model.find(*q);
		if (t != model.end())
		{
			fv.emplace_back(t->second);
		}
	}
	return fv;
}
//更新权重。
void linear_model::update_weight(const sentence  &sen, int pos,const string &max_tag, const string &correct_tag,int current_time)
{

	int max_id = tag[max_tag];
	int correct_id = tag[correct_tag];
	vector<string> f=create_feature(sen, pos);
	vector<int> fv = get_id(f);
	int last_w_value;
	int last_time;
	for (auto z = fv.begin(); z != fv.end(); z++)
	{
		//更新正确的权重
		int index = correct_id * model.size() + *z;
		last_w_value = w[index];
		w[index]++;
		//更新V;
		last_time = update_time[index]; //得到上次的更新时间。
		update_time[index]= current_time;//更新时间。
		v[index] += (current_time - last_time - 1)*last_w_value+w[index]; //更新权值。

		//更新错误的权重
		int index1 = max_id * model.size() + *z;
		last_w_value = w[index1];
		w[index1]--;
		//更新V
		last_time = update_time[index1];
		update_time[index1] = current_time;//更新时间。
		v[index1] += (current_time - last_time - 1)*last_w_value + w[index1]; //更新权值。
	}
}
int linear_model::count_score(int offset, vector<int> &fv,bool averaged)
{
	double score = 0;
	if (averaged)
	{
		for (auto z0 = fv.begin(); z0 != fv.end(); z0++)
		{
			score = score + v[offset + *z0];
		}
	}
	else
	{
		for (auto z0 = fv.begin(); z0 != fv.end(); z0++)
		{
			score = score + w[offset + *z0];
		}
	}
	return score;
}
string linear_model::maxscore_tag(const sentence  &sen, int pos,bool averaged)
{
	double max_num = -1e10, score;
	string max_tag;
	vector<string> f = create_feature(sen, pos);
	vector<int> fv = get_id(f);
	for (auto z = tag.begin(); z != tag.end(); z++)//遍历词性
	{
		int offset = z->second*model.size();
		score = count_score(offset, fv, averaged);
		if (score >max_num+1e-10 )
		{
			max_num = score;
			max_tag = z->first;
		}
	}
	return max_tag;
}
double linear_model::evaluate(dataset & data,bool averaged)
{
	int c = 0, total = 0;
	for (auto z = data.sentences.begin(); z != data.sentences.end(); z++)
	{
		for (int z0 = 0; z0 < z->word.size(); z0++)
		{
			total++;
			string max_tag = maxscore_tag(*z, z0, averaged);
			string correct_tag = z->tag[z0];
			if (max_tag == correct_tag)
			{
				c++;
			}
		}
	}
	return (c / double(total));
}
void linear_model::online_training(bool averaged, bool shuffle, int iterator, int exitor)
{
	ofstream result("small_result_shuffle.txt");
	if (!result)
	{
		cout << "don't open feature file" << endl;
	}
	result << train.name << "共" << train.sentence_count << "个句子，共" << train.word_count << "个词" << endl;
	result << dev.name << "共" << dev.sentence_count << "个句子，共" << dev.word_count << "个词" << endl;
	if (test.name.size() != 0)
	result << test.name << "共" << test.sentence_count << "个句子，共" << test.word_count << "个词" << endl;
	result << " the total number of features is " << model.size() << endl;
	int update_times = 0;
	int count = 0;
	double max_train_precision = 0;
	int max_train_iterator = 0;

	double max_dev_precision = 0;
	int max_dev_iterator = 0;

	double max_test_precision = 0;
	int max_test_iterator = 0;

	if (averaged)
	{
		cout << "using V to predict dev data..." << endl;
		result<< "using V to predict dev data..." << endl;
	}
	else
	{
		cout << "using W to predict dev data..." << endl;
		result << "using w to predict dev data..." << endl;
	}

	DWORD t1, t2, t3, t4;
	t1 = timeGetTime();
	for (int i = 0; i < iterator; i++)
	{
		result<< "iterator " << i << endl;
		cout << "iterator " << i << endl;
		t3 = timeGetTime();
		if (shuffle)
		{
			cout << "正在打乱数据" << endl;
			train.shuffle();
		}
		for (auto sen = train.sentences.begin(); sen != train.sentences.end(); ++sen)
		{
			for (int pos = 0; pos < sen->word.size(); pos++)
			{
				string max_tag = maxscore_tag(*sen, pos, false);
				string correct_tag = sen->tag[pos];
				if (max_tag != correct_tag)
				{
					update_times++;
					update_weight(*sen, pos, max_tag, correct_tag,update_times);
				}
			}
		}

		if (averaged)
		{
			int last_time, last_w_value;
			int current_time = update_times;
			for (int i = 0; i < v.size(); i++)
			{
				last_time = update_time[i];
				last_w_value = w[i];
				if (last_time != current_time)
				{
					update_time[i] = current_time;
					v[i] += (current_time - last_time-1)*last_w_value; //更新权值。
				}
			}
		}

	//	save_file(i);

		double train_precision = evaluate(train, averaged);
		cout <<train.name << "=" << train_precision << endl;
		result << train.name << "=" << train_precision << endl;

		double dev_precision = evaluate(dev, averaged);
		cout << dev.name << "=" << dev_precision << endl;
		result << dev.name << "=" << dev_precision << endl;

		if (train_precision > max_train_precision)
		{
			max_train_precision = train_precision;
			max_train_iterator = i;
		}
		if (dev_precision > max_dev_precision)
		{
			max_dev_precision = dev_precision;
			max_dev_iterator = i;
			count = 0;
		}		
		else
		{
			count++;
		}
		if (test.name.size() != 0)
		{
			double test_precision = evaluate(test, averaged);
			cout << test.name << "=" << test_precision << endl;
			result << test.name << "=" << test_precision << endl;
			if (test_precision > max_test_precision)
			{
				max_test_precision = test_precision;
				max_test_iterator = i;
			}
		}

		t4 = timeGetTime();
		cout << "Use Time:" << (t4 - t3)*1.0 / 1000 << endl;
		result << "Use Time:" << (t4 - t3)*1.0 / 1000 << endl;
		if (count >= exitor)
		{
			break;
		}
	}

	cout << train.name +"最大值是："<< "=" << max_train_precision <<"在"<<max_train_iterator<<"次"<<endl;
	cout << dev.name + "最大值是：" << "=" << max_dev_precision << "在" << max_dev_iterator << "次" << endl;

	result << train.name + "最大值是：" << "=" << max_train_precision << "在" << max_train_iterator << "次" << endl;
	result<< dev.name + "最大值是：" << "=" << max_dev_precision << "在" << max_dev_iterator << "次" << endl;
	if (test.name.size() != 0)
	{
		cout << test.name + "最大值是：" << "=" << max_test_precision << "在" << max_test_iterator << "次" << endl;
		result << test.name + "最大值是：" << "=" << max_test_precision << "在" << max_test_iterator << "次" << endl;
	}
	t2 = timeGetTime();
	cout << "Use Time:" << (t2 - t1)*1.0 / 1000 << endl;
	result<< "Use Time:" << (t2 - t1)*1.0 / 1000 << endl;
}

linear_model::linear_model(string train_, string dev_, string test_)
{
	if (train_ != "")
	{
		train.read_data(train_);
	}
	if (dev_ != "")
	{
		dev.read_data(dev_);
	}
	if (test_ != "")
	{
		test.read_data(test_);
	}
}
void linear_model::save_file(int i)
{
	string file_name = "feature" + to_string(i);
	ofstream feature_file(file_name);
	if (!feature_file)
	{
		cout << "don't open feature file" << endl;
	}
	for (auto z = value.begin(); z != value.end(); z++)
	{
		int i = (*z).find(":");
		string left = (*z).substr(0, i);
		string right = (*z).substr(i + 1);
		int f = model[*z];
		for (auto z = tag.begin(); z != tag.end(); z++)
		{
			int offset = model.size()*z->second;
			int index = offset + f;
			if (w[index] != 0)
			{
				string feature = left + z->first + right+"\t"+to_string(w[index]);
				feature_file << feature << endl;
			}
		}
	}
}
linear_model::~linear_model()
{
}
