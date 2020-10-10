class max_match:
    #根据语料建立一个字典
    def build_dict(self,data_set,dict_set):
        infile=open(data_set,"r",encoding="utf-8")
        outfile = open(dict_set, "w", encoding="utf-8")
        dit={}
        for i in infile:
            if(len(i)>1):
                value=i.split()[1]
                if(value not in dit.keys()):
                    dit[value]=1
                    outfile.write(value+"\n")
        infile.close()
        outfile.close()
    #根据给出的语料建立一个毛文本
    def build_txt(self,data_set,txt_set):
        infile=open(data_set,"r",encoding="utf-8")
        outfile=open(txt_set, "w", encoding="utf-8")
        for i in infile:
            if(len(i)>1):
                outfile.write(i.split()[1])
            else:
                outfile.write("\n")
        infile.close()
        outfile.close()
    #最大匹配分词之前向匹配
    def maxmatch_forward(self,txt_set,dict_set,forward_out_set):
        dict_infile=open(dict_set,"r",encoding="utf-8")
        max_num=0
        words_dic={}
        for i in dict_infile:
            i=i.strip()
            if(len(i)>max_num):
                max_num=len(i)
            words_dic[i]=1
        dict_infile.close()
        txt_infile = open(txt_set, "r", encoding="utf-8")
        forward_outfile=open(forward_out_set,"w",encoding="utf-8")
        for sentence in txt_infile:
            sentence=sentence.strip()
            p1=0
            while(p1<len(sentence)):
                p2=p1+max_num
                if (p2>len(sentence)):
                    p2=len(sentence)
                while(p2-p1>1 and sentence[p1:p2] not in words_dic.keys()):
                    p2-=1
                forward_outfile.write(sentence[p1:p2]+"\n")
                p1=p2
        txt_infile.close()
        forward_outfile.close()

    # 最大匹配分词之后向匹配
    def maxmatch_backword(self,txt_set,dict_set,backward_out_set):
        dict_infile=open(dict_set,"r",encoding="utf-8")
        max_num=0
        words_dic={}
        for i in dict_infile:
            i=i.strip()
            if(len(i)>max_num):
                max_num=len(i)
            words_dic[i]=1
        dict_infile.close()
        txt_infile = open(txt_set, "r", encoding="utf-8")
        backward_outfile=open(backward_out_set,"w",encoding="utf-8")
        for sentence in txt_infile:
            sentence=sentence.strip()
            p1=len(sentence)
            ls=[]
            while(p1>0):
                p2=p1-max_num
                if(p2<0):
                    p2=0
                while (p1-p2>1 and sentence[p2:p1] not in words_dic.keys()):
                    p2+=1
                ls.insert(0,sentence[p2:p1])
                p1=p2
            for i in ls:
                backward_outfile.write(i+"\n")
        txt_infile.close()
        backward_outfile.close()

    #评价程序
    def evaluate(self,data_predict,data_correct):
        predict_infile=open(data_predict,"r",encoding="utf-8")
        correct_infile=open(data_correct,"r",encoding="utf-8")
        predict_lst=[]
        correct_lst=[]
        for i in predict_infile:
            i=i.strip()
            predict_lst.append(i)
        for i in correct_infile:
            if (len(i)>1):
                correct_lst.append(i.split()[1])
        predict_infile.close()
        correct_infile.close()
        allpredict_count=len(predict_lst)#识别出的个体总数
        a=0
        b=0
        correctpredict_count=0#正确识别的词数
        allrecall_count=len(correct_lst)#测试集中存在的个体总数
        while(a<allpredict_count and b<allrecall_count):
            if(predict_lst[a]==correct_lst[b]):
                correctpredict_count+=1
            else:
                temp_predict,temp_correct=predict_lst[a],correct_lst[b]
                while(len(temp_correct)!=len(temp_predict)):
                    if(len(temp_correct)>len(temp_predict)):
                        a+=1
                        temp_predict+=predict_lst[a]
                    elif (len(temp_correct) < len(temp_predict)):
                        b+=1
                        temp_correct+=correct_lst[b]
            a+=1
            b+=1
        P=correctpredict_count/allpredict_count
        R=correctpredict_count/allrecall_count
        F=(P*R*2)/(P+R)
        print("正确识别的词数为%d\n识别出的个体总数为%d\n测试集中存在的个体总数为%d\n正确率为%f\n召回率为%f\nF值为%f\n"%(correctpredict_count,allpredict_count,allrecall_count,P,R,F))

if __name__ == '__main__':
    max_match_object=max_match()
    max_match_object.build_dict("data.conll","word_dict")
    max_match_object.build_txt("data.conll","word_txt")
    print("正向匹配分词结果为：")
    max_match_object.maxmatch_forward("word_txt","word_dict","forward_out")
    max_match_object.evaluate("forward_out","data.conll")
    print("反向匹配分词结果为：")
    max_match_object.maxmatch_backword("word_txt", "word_dict", "backward_out")
    max_match_object.evaluate("backward_out", "data.conll")






