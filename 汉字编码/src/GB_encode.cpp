#include<stdio.h>
#include<process.h>
 /*此程序用于识别GB编码的汉字和其他单独的字符，汉字占两个字节，
 其他独立的字符占一个字节,判断是汉字还是单独的字符，
 只需要从左向右扫描第一个字节的第一位，如果是0，则表示是独立的字符，如果是1，则是汉字*/ 
main(){
	FILE *fp;
	char ch[2];
	char a;
	fp=fopen("GB.txt","r");//文件名称为GB.txt 
	a=fgetc(fp);
	int count=0;
	while(a!=EOF){//是ASCII字符 
		char b = a;
		int t=(b>>7)&1;
		if(t==0){
		count++;
		printf("%c ",a);
		a=fgetc(fp);
		}
		if(t==1){//是汉字 
			ch[0]=a;
			a=fgetc(fp);
			ch[1]=a;
			count++;
			printf("%s ",ch);
			a=getc(fp);
		}
		if((a>=-1&&a<48)||a>122){
			a=fgetc(fp); 
			count++;
		}
	}
	fclose(fp);
	printf("\n共有character：%d个",count);
} 
