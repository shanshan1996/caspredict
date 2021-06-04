
import sys
from Bio import SeqIO


AAs=list('ACDEFGHIKLMNPQRSTVWY')
DoubleAAs=[i+j for i in AAs for j in AAs]
topDoubleAAs=[i+j for i in AAs for j in AAs]

ParameterNum=len(sys.argv)

for n in range(0,400,1):
    fout=open(str(n+1),'w')

    for pn in range(1,ParameterNum,1):
        fin=open(sys.argv[pn],'r')

        for record in SeqIO.parse(fin,'fasta'):
            pep=str(record.seq)
            DiAA_fre={}
            for DiAA in DoubleAAs:
                DiAA_fre[DiAA]=0

            for i in range(0,len(pep)-1,1):
                try:
                    DiAA_fre[pep[i:i+2]]+=1
                except:
                    print(record)
    
            fout.write(str(pn)+" ")
    
            j=1
            DiAA_num=float(len(pep)-1)
            for DiAA in topDoubleAAs[0:n+1]:
                DiAA_fre[DiAA]/=DiAA_num
                fout.write(str(j)+":")
                fout.write(str(DiAA_fre[DiAA]))
                fout.write(' ')
                j+=1
  
            fout.write('\n')

        fin.close()
    
    
    fout.close()
