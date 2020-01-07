# -*- coding: utf-8 -*-

if __name__ == "__main__":
    fin = open("./data/data.conll", "r")
    fout = open("./data/data.txt", "w")
    while True:
        line = fin.readline()
        if not line:
            break
        if (line == "\n"):
            fout.write("\n")
        else:
            L = line.split()
            fout.write(L[1])
            # L=list(filter(lambda x:x!="_",L))
    fin.close()
    fout.close()
