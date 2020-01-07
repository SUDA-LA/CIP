//
// Created by Jacob_Zhou on 2018/10/6.
//

#include <stdio.h>

int main(){
    FILE *fp = NULL;
    FILE *output = NULL;
    char buff[4];
    buff[3] = 0;
    int buff_size = 65535;
    int now_char = 0;
    int bit_110 = 3 << 6;
    int bit_mark_110 = 7 << 5;
    int bit_1110 = 7 << 5;
    int bit_mark_1110 = 15 << 4;
    int char_count = 0;

    fp = fopen("..\\data\\UTF-8.txt", "r");
    output = fopen("..\\data\\UTF-8_out.txt", "w");
    now_char = fgetc(fp);
    if(fp == NULL || output == NULL){
        return -1;
    }
    while(now_char != EOF){
        if((now_char & bit_mark_110) == bit_110){
            buff[0] = (char)now_char;
            buff[1] = (char)fgetc(fp);
            buff[2] = 0;
            printf("%s ", buff);
            fprintf(output, "%s ", buff);
        } else if((now_char & bit_mark_1110) == bit_1110) {
            buff[0] = (char)now_char;
            buff[1] = (char)fgetc(fp);
            buff[2] = (char)fgetc(fp);
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