//
// Created by Jacob_Zhou on 2018/10/6.
//

#include <stdio.h>

int main(){
    FILE *fp = NULL;
    FILE *output = NULL;
    char buff[3];
    buff[2] = 0;
    int buff_size = 65535;
    int now_char = 0;
    int bit_mark = 1 << 7;
    int char_count = 0;

    fp = fopen("..\\data\\GBK.txt", "r");
    output = fopen("..\\data\\GBK_out.txt", "w");
    if(fp == NULL || output == NULL){
        return -1;
    }
    now_char = fgetc(fp);
    while(now_char != EOF){
        if((now_char & bit_mark) == bit_mark){
            buff[0] = (char)now_char;
            buff[1] = (char)fgetc(fp);
            printf("%s ", buff);
            fprintf(output, "%s ", buff);
        } else {
            printf("%c ", now_char);
            fprintf(output, "%c ", now_char);
        }
        char_count++;
        now_char = fgetc(fp);
    }
    printf("\nchar count: %d", char_count);
    fprintf(output, "\nchar count: %d", char_count);
}