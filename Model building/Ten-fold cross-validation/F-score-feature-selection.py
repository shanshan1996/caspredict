#The program in LIBSVM are used in this script
#Copyright (c) 2000-2021 Chih-Chung Chang and Chih-Jen Lin

import random
import numpy
import sys
from sys import argv
from svmutil import *
from random import randrange , seed
import os
from os import system
from os import unlink
from subprocess import *
from functools import reduce
##### Path Setting #####

is_win32 = (sys.platform == 'win32')
if not is_win32:
	gridpy_exe = "./grid.py -log2c -2,9,2 -log2g 1,-11,-2"
	svmtrain_exe="../svm-train"
	svmpredict_exe="../svm-predict"
else:
	gridpy_exe = r"F:\peerj\reviseV1\10-fold\libsvm-3.24\tools\grid.py -log2c -2,6,2 -log2g 1,-8,-2"
	svmtrain_exe=r"F:\peerj\reviseV1\10-fold\libsvm-3.24\windows\svm-train.exe"
	svmpredict_exe=r"F:\peerj\reviseV1\10-fold\libsvm-3.24\windows\svm-predict.exe"
###return a dict containing F_j

def cal_Fscore(labels,samples):

    data_num=float(len(samples))
    p_num = {} #key: label;  value: data num
    sum_f = [] #index: feat_idx;  value: sum
    sum_l_f = {} #dict of lists.  key1: label; index2: feat_idx; value: sum
    sumq_l_f = {} #dict of lists.  key1: label; index2: feat_idx; value: sum of square
    F={} #key: feat_idx;  valud: fscore
    max_idx = -1

    ### pass 1: check number of each class and max index of features
    for p in range(len(samples)): # for every data point
        label=labels[p]
        point=samples[p]

        if label in p_num: p_num[label] += 1
        else: p_num[label] = 1

        for f in point.keys(): # for every feature
            if f>max_idx: max_idx=f
    ### now p_num and max_idx are set

    ### initialize variables
    sum_f = [0 for i in range(max_idx)]
    for la in p_num.keys():
        sum_l_f[la] = [0 for i in range(max_idx)]
        sumq_l_f[la] = [0 for i in range(max_idx)]

    ### pass 2: calculate some stats of data
    for p in range(len(samples)): # for every data point
        point=samples[p]
        label=labels[p]
        for tuple in point.items(): # for every feature
            f = tuple[0]-1 # feat index
            v = tuple[1] # feat value
            sum_f[f] += v
            sum_l_f[label][f] += v
            sumq_l_f[label][f] += v**2
    ### now sum_f, sum_l_f, sumq_l_f are done

    ### for each feature, calculate f-score
    eps = 1e-12
    for f in range(max_idx):
        SB = 0
        for la in p_num.keys():
            SB += (p_num[la] * (sum_l_f[la][f]/p_num[la] - sum_f[f]/data_num)**2 )

        SW = eps
        for la in p_num.keys():
            SW += (sumq_l_f[la][f] - (sum_l_f[la][f]**2)/p_num[la]) 

        F[f+1] = SB / SW

    return F

def value_cmpf(x):
    return (-x[1]);

def feat_num_try(f_tuple):
    for i in range(len(f_tuple)):
        if f_tuple[i][1] < 1e-20:
            i=i-1; break
    #only take first eight numbers (>1%)
    return feat_num_try_half(i+1)[:i+1]

def feat_num_try_half(max_index):
    v=[]
    b=10
    while b <= max_index:
        v.append(b)
        b=b+10
    return v

def writedata(samples,labels,filename):
    fp=sys.stdout
    if filename:
        fp=open(filename,"w")

    num=len(samples)
    for i in range(num):
        if labels: 
            fp.write("%s"%labels[i])
        else:
            fp.write("0")
        kk=list(samples[i].keys())
        kk.sort()
        for k in kk:
            fp.write(" %d:%.10f"%(k,samples[i][k]))
        fp.write("\n")

    fp.flush()
    fp.close()

def select(sample, feat_v):
    new_samp = []

    #for each sample
    for s in sample:
        point={}
        i=1
        #for each feature to select
        for f in feat_v:
            if f in s: point[i]=s[f]
            i=i+1

        new_samp.append(point)

    return new_samp

def train_svm(tr_file):
    cmd = "python %s %s" % (gridpy_exe,tr_file)
    print(cmd)
    print('Cross validation...')
    std_out = Popen(cmd, shell = True, stdout = PIPE).stdout

    line = ''
    while 1:
        last_line = line
        line = std_out.readline()
        if not line: break
    c,g,rate = map(float,last_line.split())

    print('Best c=%s, g=%s CV rate=%s' % (c,g,rate))

    return c,g,rate

def rem_file(filename):
    #system("rm -f %s"%filename)
    unlink(filename)

def cal_acc(pred_y, real_y):
    p1=0
    tp=0
    tn=0
    for i in range(len(pred_y)):
        if real_y[i]==1:
           p1=p1+1 #positive samples
           if real_y[i]==pred_y[i]:
              tp=tp+1 #true positive
        else:
           if real_y[i]==pred_y[i]:
              tn=tn+1 #true negative
    n1=len(pred_y)-p1 #negative samples
    fp=n1-tn #false positive
    fn=p1-tp #false negative
    sn=tp/(tp+fn)
    sp=tn/(tn+fp)
    acc=(tp+tn)/(tp+fn+fp+tn)
    mcc=(tp*tn-fp*fn)/numpy.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    print("ACC: %d/%d"%(tp+tn, len(pred_y)))
    return sn,sp,acc,mcc

AAs=list('ACDEFGHIKLMNPQRSTVWY')
DoubleAAs=[i+j for i in AAs for j in AAs]
train_file = argv[-1]
nr_fold = 10
features=[]
result_sn=[]
result_sp=[]
result_acc=[]
result_mcc=[]
p=1
x=1
while(p<2):
#shuffle data
    total_sn=[]
    total_sp=[]
    total_acc=[]
    total_mcc=[]
    shuffle_i1=[]
    shuffle_j1=[]
    prob_y, prob_x = svm_read_problem(train_file)
    prob_l = len(prob_y)
    for i in range(0,int(prob_l/2),1):
        j = randrange(0,int(prob_l/2))
        shuffle_i1.append(i)
        shuffle_j1.append(j)
        prob_x[i], prob_x[j] = prob_x[j], prob_x[i]
        prob_y[i], prob_y[j] = prob_y[j], prob_y[i]
    shuffle_i2=[]
    shuffle_j2=[]
    for i in range(int(prob_l/2),int(prob_l),1):
        j = randrange(int(prob_l/2),int(prob_l))
        shuffle_i2.append(i)
        shuffle_j2.append(j)
        prob_x[i], prob_x[j] = prob_x[j], prob_x[i]
        prob_y[i], prob_y[j] = prob_y[j], prob_y[i]
    j=int(prob_l/2)
    for i in range(0,int(prob_l/2),2):
        prob_x[i], prob_x[j] = prob_x[j], prob_x[i]
        prob_y[i], prob_y[j] = prob_y[j], prob_y[i]
        j=j+2
    sd=open('shuffle_seed.txt',"a")
    sd.write("%s\n%s\n%s\n%s\n"%(shuffle_i1,shuffle_j1,shuffle_i2,shuffle_j2))
    sd.close()
    shuffle_train_name = "shuffle_train"+str(p)
    writedata(prob_x,prob_y,shuffle_train_name)
#cross training : folding
    for k in range(nr_fold):
        begin = k * prob_l // nr_fold
        end = (k + 1) * prob_l // nr_fold
        train_x = prob_x[:begin] + prob_x[end:]
        train_y = prob_y[:begin] + prob_y[end:]
        test_x = prob_x[begin:end]
        test_y = prob_y[begin:end]

        score_dict=cal_Fscore(train_y,train_x)
        score_tuples = list(score_dict.items())
        score_tuples.sort(key = value_cmpf)
        feat_v = score_tuples
        for i in range(len(feat_v)): feat_v[i]=score_tuples[i][0]
        ###write (sorted) f-score list in another file
        f_tuples = list(score_dict.items())
        f_tuples.sort(key = value_cmpf)
        fd = open("%s.fscore"%train_file, 'w')
        for t in f_tuples:
            fd.write("%d:%.6f\n"%t)
        fd.close()
        ### decide sizes of features to try
        fnum_v = feat_num_try(f_tuples)
        est_acc=[]
        est_c=[]
        est_g=[]
        #for each possible feature subset
        for j in range(len(fnum_v)):

            fn = fnum_v[j]  # fn is the number of features selected
            fv = feat_v[:fn] # fv is indices of selected features

            #pick features
            tr_sel_samp = select(train_x, fv)
            tr_sel_name = train_file+".train"+str(j+1)

            writedata(tr_sel_samp,train_y,tr_sel_name)
            command1 = "D:\\installpath\\libsvm-3.24\\windows\\svm-scale.exe " + tr_sel_name + " > " + tr_sel_name +".scale"
            os.system(command1)
            scale_name=tr_sel_name+".scale"
            print(j+1)
            #choose best c, gamma from splitted training sample
            c,g, cv_acc = train_svm(scale_name)
            est_c.append(c)
            est_g.append(g)
            est_acc.append(cv_acc)
            rem_file(tr_sel_name)
            rem_file("%s.scale"%tr_sel_name)
            rem_file("%s.out"%scale_name)
        fnum=fnum_v[[i for i,x in enumerate(est_acc) if x==max(est_acc)][-1]]
        sel_fv = feat_v[:fnum]
        features.append(sel_fv)
        h=fnum_v.index(fnum)
        best_c=est_c[h]
        best_g=est_g[h]

        tp_sel_samp = select(train_x, sel_fv)
        tp_sel_name = train_file+".tr"+str(k+x)
        writedata(tp_sel_samp,train_y,tp_sel_name)
        range_name=tp_sel_name+".range"
        command2 = "D:\\installpath\\libsvm-3.24\\windows\\svm-scale.exe " + "-s "+range_name+" "+tp_sel_name + " > " + tp_sel_name +".scale"
        os.system(command2)
        command3 = "F:\\peerj\\reviseV1\\10-fold\\libsvm-3.24\\windows\\svm-train.exe "+"-b 1 -c "+str(best_c)+" -g "+str(best_g)+" " + tp_sel_name +".scale" + " "+str(tp_sel_name)+".model"
        os.system(command3)
### do testing 
    #picking features
        test_sel_samp = select(test_x, sel_fv)
        te_sel_name = train_file+".te"+str((k+x))
        out_file = "%s.o"%te_sel_name+str((k+x))
        writedata(test_sel_samp,test_y,te_sel_name)
        command4 = "D:\\installpath\\libsvm-3.24\\windows\\svm-scale.exe " +"-r "+range_name+" "+ te_sel_name + " > " + te_sel_name +".scale"
        os.system(command4)
        command5 = "F:\\peerj\\reviseV1\\10-fold\\libsvm-3.24\\windows\\svm-predict.exe "+te_sel_name +".scale"+" "+str(tp_sel_name)+".model"+" "+str(out_file)
        os.system(command5)
        pred_y=[]
        fp = open(out_file)
        line = fp.readline()
        while line:
            pred_y.append( float(line) )
            line = fp.readline()
        fp.close()
        sn,sp,acc,mcc = cal_acc(pred_y, test_y)
        total_sn.append(sn)
        total_sp.append(sp)
        total_acc.append(acc)
        total_mcc.append(mcc)
        
        h_mm=[]
        for m8 in range(len(sel_fv)):
            w8=DoubleAAs[sel_fv[m8]-1]
            h_mm.append(w8)
            
        result_total=open('Allfeature_Result.txt',"a")
        result_total.write('features:%s\n best c and g:%s %s acc:%.6f\n DPC-features:%s\n size:%s\n' % (sel_fv, best_c, best_g, acc, h_mm, len(sel_fv)))
        result_total.close()
        rem_file(tp_sel_name)
        rem_file("%s.model"%tp_sel_name)
        rem_file("%s.range"%tp_sel_name)
        rem_file("%s.scale"%tp_sel_name)
        rem_file("%s.scale"%te_sel_name)
        rem_file(te_sel_name)
        rem_file(out_file)
        print(k+1)
    sum_sn=0
    sum_sp=0
    sum_acc=0
    sum_mcc=0

    for w in range(len(total_acc)):
        sum_sn=sum_sn+total_sn[w]
        sum_sp=sum_sp+total_sp[w]
        sum_acc=sum_acc+total_acc[w]
        sum_mcc=sum_mcc+total_mcc[w]
    average_sn=sum_sn/len(total_sn)
    average_sp=sum_sp/len(total_sp)
    average_acc=sum_acc/len(total_acc)
    average_mcc=sum_mcc/len(total_mcc)
    result_sn.append(average_sn)
    result_sp.append(average_sp)
    result_acc.append(average_acc)
    result_mcc.append(average_mcc)
    x=x+nr_fold
    p=p+1
print(result_acc)
shuffle_sum_sn=reduce(lambda x,y:x+y,result_sn)
shuffle_sum_sp=reduce(lambda x,y:x+y,result_sp)
shuffle_sum_acc=reduce(lambda x,y:x+y,result_acc)
shuffle_sum_mcc=reduce(lambda x,y:x+y,result_mcc)
shuffle_avrage_sn=shuffle_sum_sn/len(result_sn)
shuffle_avrage_sp=shuffle_sum_sp/len(result_sp)
shuffle_avrage_acc=shuffle_sum_acc/len(result_acc)
shuffle_avrage_mcc=shuffle_sum_mcc/len(result_mcc)
q=[i for i,x in enumerate(result_acc) if x==max(result_acc)][-1]
a=(q+1)*nr_fold-nr_fold
b=(q+1)*nr_fold
print(a)
print(b)
last_feature=features[a:b]
sets=[]
for n in range(len(last_feature)):
    sets.append(set(last_feature[n]))

fo = open("feature_sets", "w")
fo.write("%s\n"%sets)
fo.write("a=%s b=%s\n"%(a,b))
fo.write("average_sn:%s\n average_sp:%s\n average_acc:%s\n average_mcc:%s\n"%(result_sn,result_sp,result_acc,result_mcc))
fo.write("max_average_acc:%s\n"%max(result_acc))
fo.write("shuffle_avrage_sn:%s\n shuffle_avrage_sp:%s\n shuffle_avrage_acc:%s\n shuffle_avrage_mcc:%s\n"%(shuffle_avrage_sn,shuffle_avrage_sp,shuffle_avrage_acc,shuffle_avrage_mcc))
fo.close()
