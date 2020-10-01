#include "pch.h"
#include <iostream>
#include <string>

using namespace std;


int main()
{
	string s;
	while (cin >> s) {
		int count = 0;
		int i = 0;
		while (i < s.size()) {
			if ((s[i] & 0b10000000) == 0b10000000) {
				cout << s[i] << s[i + 1] << " ";
				i += 2;
				count++;
			}
			else {
				cout << s[i]<<" ";
				i += 1;
				count++;
			}
		}
		cout << count << endl;
	}
	//char c[3] = { 0,0,0 };
	//while (cin>>c) {
	//	c[0] -= 0xA0;
	//	c[1] -= 0xA0;
	//	cout << c[0] - 0x0 << endl;
	//	cout << c[1] - 0x0 << endl;
	//}
	//string s = "中华人民共和国";
	//for (int i = 0; i < s.size(); i += 2) {
	//	cout << i << s[i] << s[i + 1] << endl;
	//}
	//string s_c = "这是一个 测试！";
	//string s_e = "this is an attempt!";
	//cout << s_c.size() << strlen(s_c.c_str())<< endl;
	//cout << s_e.size() << endl;
	return 0;

}


