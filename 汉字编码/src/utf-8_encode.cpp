#include <stdio.h>
/*此程序用于识别UTF-8编码的汉字和其他单独的字符，UTF-8用1-6字节表示一个字符，我们需要从左往右进行扫描
根据当前字节的前几位来决定当前字符由多少个字节组成，分别有六种情况，对应的字节数从6到1递减，从而可以分出所有的字来*/ 
main()
{
    FILE *fp = fopen("utf-8.txt","r");//测试集名称为123.txt 
    char ch = getc(fp);
    int count=0;
    while (ch!= EOF)
    {
        int test = ch & 0xff;
        if(test>=0xfc)//一个字由六个字节组成的情况 
		{
			printf("%c",ch);
            for(int i=0 ; i<5 ; i++)
            {
                ch = getc(fp);
                printf("%c",ch);
            }
            printf("  ");
            count++;
            ch = getc(fp);
            continue;
        	
		}
		else if(test >= 0xf8 && test<0xfc)//一个字由五个字节组成的情况
		{
			printf("%c",ch);
            for(int i=0 ; i<4 ; i++)
            {
                ch = getc(fp);
                printf("%c",ch);
            }
            printf("  ");
            count++;
            ch = getc(fp);
            continue;
		}
        else if(test>=0xf0 && test<0xf8)//一个字由四个字节组成的情况
        {
            printf("%c",ch);
            for(int i=0 ; i<3 ; i++)
            {
                ch = getc(fp);
                printf("%c",ch);
            }
            printf("  ");
            count++;
            ch = getc(fp);
            continue;
        }
        else if(test >= 0xe0 && test < 0xf0)//一个字由三个字节组成的情况
		{
            printf("%c",ch);
            for(int i=0 ; i<2 ; i++)
            {
                ch = getc(fp);
                printf("%c",ch);
            }
            printf("  ");
            count++;
            ch = getc(fp);
            continue;
        }
        else if(test >= 0xc0 && test <0xe0)//一个字由两个字节组成的情况
		{
            printf("%c",ch);
            ch = fgetc(fp);
            printf("%c  ",ch);
            count++;
            ch = fgetc(fp);
            continue;
        }
        else//一个字由一个字节组成的情况
        {
            printf("%c  ",ch);
            count++;
            ch = getc(fp);
            continue;
        }
    }
    printf("\n");
    printf("%d",count);
    fclose(fp);
}
