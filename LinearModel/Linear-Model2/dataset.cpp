#include "dataset.h"
#include<random>
void dataset::read_data(const string & file_name)
{
	name = file_name;
	ifstream file(file_name);
	if (!file)
	{
		cout << "don't open " + file_name << endl;
	}
	string line;
	sentence sen;
	while (getline(file, line))
	{
		if (line.size() == 0)
		{
			sentences.push_back(sen);
			sen.~sentence();
			sentence_count++;
			continue;
		}
		word_count++;
		int t0 = line.find("\t") + 1;
		int t1 = line.find("\t", t0);
		string word = line.substr(t0, t1 - t0);
		int t2 = line.find("\t", t1 + 1) + 1;
		int t3 = line.find("\t", t2);
		string tag = line.substr(t2, t3 - t2);
		sen.word.push_back(word);
		sen.tag.push_back(tag);
		//构建词的单个元素。
		vector<string> word_char;
		for (unsigned t4 = 0; t4 < word.size();)
		{
			if ((unsigned char)word[t4] > 129 && (unsigned char)word[t4 + 1] > 64)
			{
				word_char.emplace_back(word.substr(t4, 2));
				t4 = t4 + 2;
			}
			else
			{
				word_char.emplace_back(word.substr(t4, 1));
				t4++;
			}
		}
		sen.word_char.emplace_back(word_char);
	}
	cout << name << "contains sentence count " << sentence_count << endl;
	cout << name << "contains word count " << word_count << endl;
}
#include<ctime>
#include<unordered_set>
#include "time.h"
#include"windows.h"
void dataset::shuffle()
{
	default_random_engine t(timeGetTime());

	uniform_int_distribution<int> u(0, sentences.size()-1);

	vector<sentence> sentences_;
	unordered_set<int> unique;
	int j = 0;
	while (unique.size() != sentences.size())
	{
		j++;
		int i = u(t);
		if (unique.find(i)==unique.end())
		{
			sentences_.emplace_back(sentences[i]);
			unique.insert(i);
		}
	}
	sentences = sentences_;
}

dataset::dataset()
{
	



}


dataset::~dataset()
{
}
