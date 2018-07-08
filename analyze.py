#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 09:05:18 2018

# Created by arlencai on 2018/6/15.
# Copyright © 2018 jinqianhe. All rights reserved.

draw counfusion matrix, ROC, PRC for multi-class classification
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools


# ==================== load data  ===================
"""
can be customized
"""
from sklearn.externals import joblib
# load data
pred_result_dict = joblib.load('pred_result')
pred_score_np,label_int_np = pred_result_dict.values()

# define class names
class_names = ['remission','hypo-mania','mania']

# ================ data preprocessing =======================
# softmax the probabilistic predictions
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum() # only difference

pred_softmax_np = np.apply_along_axis(lambda x: softmax(x),1,pred_score_np)

# convert prob result to int result
pred_int_np = np.argmax(pred_softmax_np, axis=1)

# ==================== plot counfusion matrix ================
"""
Plot normalized confusion matrix
"""
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(label_int_np, pred_int_np)
#np.set_printoptions(precision=28)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()


# =================== plot ROC curve ================
"""
Compute ROC curve and ROC area for each class
"""
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc
enc = OneHotEncoder(sparse=False)
enc.fit(label_int_np.reshape(-1,1))
onehot=enc.transform(label_int_np.reshape(-1,1))

fpr = dict()
tpr = dict()
roc_auc = dict()
thresholds = dict()
for (i, class_name) in enumerate(class_names):
    fpr[class_name], tpr[class_name], thresholds[class_name] = roc_curve(onehot[:, i], pred_softmax_np[:, i])
    roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])
    
    youdens = tpr[class_name] - fpr[class_name]
    index=np.argmax(youdens)
    #youden[class_name]=tpr[class_name](index) 
    fpr_val=fpr[class_name][index]
    tpr_val=tpr[class_name][index]
    thresholds_val=thresholds[class_name][index]

    p_auto=pred_softmax_np[:, i].copy()
    t_auto=onehot[:, i].copy()
    p_auto[p_auto>=thresholds_val]=1
    p_auto[p_auto<thresholds_val]=0
    acc=np.float(np.sum(t_auto==p_auto))/t_auto.size
    
    
    plt.figure()
    plt.plot(fpr[class_name], tpr[class_name], color='darkorange',
         lw=2, label=class_name+ '(%0.2f)' % roc_auc[class_name])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of %s ' % class_name+ '(AUC={:.2f}, Thr={:.2}, Acc={:.2f}%'.format(
            roc_auc[class_name],thresholds_val,acc*100))
    plt.savefig('roc_%s.png'%class_name)
    plt.show()

# =================== plot PRC curve ================
"""
Compute PRC curve and recall rate on important points for each class
"""
from sklearn.metrics import precision_recall_curve
precision = dict()
recall = dict()
for (i, class_name) in enumerate(class_names):
    plt.figure()
    precision[class_name], recall[class_name], _ = precision_recall_curve(onehot[:, i], pred_softmax_np[:, i])
    call95=np.max(recall[class_name][precision[class_name]>=0.95])
    call90=np.max(recall[class_name][precision[class_name]>=0.90])
    call85=np.max(recall[class_name][precision[class_name]>=0.85])
    plt.step(recall[class_name], precision[class_name], color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall[class_name], precision[class_name], step='post', alpha=0.2,
                 color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('PRC of %s ' % class_name + '({:.2f}%, {:.2f}%, {:.2f}%)'.format(call95*100,call90*100,call85*100))
    plt.savefig('prc_%s.png'%class_name)
    plt.show()

# ================== save fig results (e.g.) =============
"""
fig results can be save as pdf format for latex writing
"""
fig, ax = plt.subplots(figsize=(15,10),dpi=300)
topic_exist_df.apply(lambda x: sum(x)/218).plot(kind='bar',
                    ax=ax,rot=0)

#设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=30)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]


#设置横纵坐标的名称以及对应字体格式
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 30,
}

plt.xlabel('Topic Ind.',font2)
plt.ylabel('Cover Rate',font2)
#plt.subplots_adjust(left=0.08,right=0.01,bottom=0)
#plt.margins(0,0)
plt.savefig('topic_cover_rate.pdf',bbox_inches='tight')
plt.show()



