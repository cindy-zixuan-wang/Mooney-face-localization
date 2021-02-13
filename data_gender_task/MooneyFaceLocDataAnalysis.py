#!/usr/bin/env python
# coding: utf-8

# # Load Data and PKGs

# ## Pkgs

# In[1]:


from os import getcwd as gwd
from os import chdir as cd
from itertools import combinations 
from scipy.stats import pearsonr as pr
from scipy.stats import zscore as zscore
from scipy.stats import ttest_rel as ttest
from scipy.stats import t as student_t
from scipy.optimize import curve_fit
from scipy.special import i0, gamma
import copy
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rpy2
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')
np.random.seed(223)
subjs = ['SL','JX_gender_2','ZW_gender_2','KP'] #['JX','ZW']
realEcc = 0.36
# analysisInput = 4 # 1 for angular errors, 2 for RT, 3 for distortion indices, 4 for eccentricity errors


# ## Global Util Func

# In[2]:


# Replace outliers by mean
def clrOutlier(data):
    mu = np.mean(data,axis=None)
    sd = np.std(data,axis=None)
    ind1 = np.nonzero(data<float(mu-3*sd))
    if len(ind1) != 0:
        data[ind1] = mu
    ind2 = np.nonzero(data>float(mu+3*sd))
    if len(ind2) != 0:
        data[ind2] = mu
    return data

def calSlope(avgHalf,win=7):
    last = len(avgHalf)-1
    shiftErr = pd.Series(avgHalf[last]).append(avgHalf[0:last])
    shiftErr.index = np.arange(last+1)
    slope = avgHalf-shiftErr
    return circ_moving_average(slope,win)

def circ_moving_average(a, n=7,circ=True) :
    if circ:
        a1 = a[-(n-1)//2:].append(a).append(a[:(n-1)//2])
    else:
        a1 = a
    a1 = np.array(a1)
    ret = np.cumsum(a1)
    ret[n:] = ret[n:] - ret[:-n]
    return pd.Series(ret[n - 1:] / n)

# # generate random indices for split half
# def genRandInd(reps=8,halfOrNot = True,subjNum = 1):
#     indices = None
#     angNum = 48
#     for ss in range(subjNum):
#         for angle in range(angNum):
#             if halfOrNot:
#                 randInd = np.random.choice(reps,reps//2,replace=False)+angle*reps+ss*angNum*reps
#             else:
#                 randInd = np.random.choice(reps,reps,replace=False)+angle*reps+ss*angNum*reps
#             if indices is not None:
#                 indices = np.hstack((indices,randInd))
#             else:
#                 indices = randInd       
#     return np.array(indices)

# # average each half
# def avgHalf(df,indices,coi):
#     half1 = df.loc[indices,:]
#     half2 = df.drop(indices)
#     resultDF = pd.DataFrame()
#     resultDF['half1'] = np.array(half1.groupby(by='locs')[coi].mean())
#     resultDF['half2']  = np.array(half2.groupby(by='locs')[coi].mean())
#     return resultDF

# split two halves based on locs
def splitData(df,coi,std=False):
    dfNew = pd.DataFrame()
    half1 = df.groupby('locs',as_index=False).sample(frac=0.5)[['locs',coi]]
    half2 = df.drop(half1.index)
    if not std:
        half1 = np.array(half1.groupby("locs")[coi].mean())
        half2 = np.array(half2.groupby("locs")[coi].mean())
    else:
        half1 = np.array(half1.groupby("locs")[coi].agg(np.std))
        half2 = np.array(half2.groupby("locs")[coi].agg(np.std))
    dfNew['half1'] = half1
    dfNew['half2']  = half2
    return dfNew

# morph the location difference
def recenter(x, threshold=24):
    for i in range(len(x)):
        if x[i] > threshold - 1:
            x[i] = x[i] - 2 * threshold
        elif x[i] < -threshold:
            x[i] = x[i] + 2 * threshold
    return x

# shift a half for a random phase
# half must be a numpy array
def shiftHalf(half):
    randPhase = np.int(np.random.choice(len(half),1))
    if randPhase == 0:
        return half
    else:
        if randPhase == len(half)-1:
            half = np.concatenate(([half[randPhase]],half[range(randPhase)]))
            return half
        else:
            half = np.concatenate((half[range(randPhase,np.int(len(half)))],half[range(randPhase)]))
            return half


# ## Data Cleaning

# ### Load Data

# In[3]:


dataAll = dict()
for subjName in subjs:
    data = pd.read_csv(('data_gender_task/'+subjName+'.csv'),header=None,                       names=['gender_RT','gender_response','real_x','real_y',                             'x','y','loc_RT','image_num','cond'],                       usecols=[11,12,18,19,33,34,38,39,45])
    data = data[-pd.isnull(data.x)]
    data = data[1:(len(data)+1)]
    data.index = range(data.shape[0])
    data = data.replace({'F':0,'M':1})
    data = data.astype('float')
    dataAll[subjName] = data


# In[4]:


dataAll['JX_gender_2'].shape


# ### Functions for Preprocessing

# In[5]:


# calculate angular response
def calAng(x,y):
    angResp = np.rad2deg(np.arctan2(y,x))
    angResp[angResp < 0] = 360+angResp[angResp < 0]
    return angResp

# calculate distance from center of the screen
def calDistance(x,y):
    return np.sqrt(x*x+y*y)

# Morph errors
def morphErr(err):
    err[err > 180] = -360 + err[err > 180]
    err[err < (-180)] = 360 + err[err < (-180)]
    return err

def calAngErr(data):
    errs = pd.DataFrame()
    errs['ang_error'] = morphErr(calAng(data.x,data.y)-                                np.round(calAng(data.real_x,data.real_y)/7.5)*7.5)
    return errs
    
# One-hot encoding
def dummy(cond):
    dummy_df = pd.DataFrame()
    dummy_df['gender'] = 1-np.mod(cond,2) #0: female; 1: male
    dummy_df['orientation'] = (cond<5).astype(int) #0: inverted; 1: upright
    dummy_df['holistic'] = (1-np.mod(np.round(cond/2),2)).astype(int) #0: low; 1: high
    return dummy_df


# ### Preprocessing

# In[6]:


def preprocessAngErr(dataAll):
    errAll = pd.DataFrame()
    for subj in subjs:   
        errs = calAngErr(dataAll[subj])
        errs['ecc_error'] = calDistance(dataAll[subj].x,dataAll[subj].y)- realEcc
        errs['locs'] = np.round(calAng(dataAll[subj].real_x,dataAll[subj].real_y)/7.5)*7.5
        errs[['gender','orientation','holistic']] = dummy(dataAll[subj].cond)
        errs['gender_acc'] = (dataAll[subj].gender_response==errs.gender).astype('int')
        errs[['gender_RT','loc_RT']] = dataAll[subj][['gender_RT','loc_RT']]
        errs['subj'] = subj
        errAll = pd.concat([errAll,errs])
    errAll.index = range(errAll.shape[0])
    return errAll
errAll = preprocessAngErr(dataAll)


# In[7]:


display(errAll.head(),errAll.tail())


# ### EDA

# #### average accuracy in gender task

# In[8]:


errAll.pivot_table(values='gender_acc',columns='orientation',aggfunc=np.mean)


# In[9]:


sns.barplot(x='orientation',y='gender_acc',data=errAll)
plt.xticks([0,1],['Inverted','Upright'])
plt.xlabel(None)
plt.ylabel('Gender Task Accuracy')
plt.title('Gender Task Performance')
plt.savefig('gender_task_acc.png')


# In[10]:


sns.barplot(x='orientation',y='gender_acc',hue='holistic',data=errAll)
plt.xticks([0,1],['Inverted','Upright'])
plt.xlabel(None)
plt.ylabel('Gender Task Accuracy')
plt.title('Gender Task Performance')
# plt.savefig('gender_task_acc.png')


# ####  distribution for raw localization errors

# In[16]:


fig, ax = plt.subplots(2,2,figsize=(12,6))
cond = 1
if cond == 0:
    condition = "orientation"
    cond1 = 'Upright'
    cond2 = 'Inverted'
elif cond == 1:
    condition = "gender_acc"
    cond1 = 'Hit'
    cond2 = 'Miss'
else:
    condition = "holistic"
    cond1 = 'High-holistic'
    cond2 = 'Low-holistic'
subjIter = 0
for subj in subjs:
    sns.violinplot(y=errAll[errAll['subj']==subj]['ang_error'],                x = errAll[errAll['subj']==subj][condition],                ax=ax[int(subjIter/2)][int(np.mod(subjIter,2))],                )
    ax[int(subjIter/2)][int(np.mod(subjIter,2))].set_xticklabels([cond2,cond1])
    ax[int(subjIter/2)][int(np.mod(subjIter,2))].set_xlabel('')
    subjIter += 1
ax[int(len(subjs)/2-1)][0].set_xlabel('Localization errors (deg)')
plt.suptitle('Distribution of Localization Errors for '+cond1+' vs. '+cond2+' Faces')
plt.savefig('distribution_raw_err_'+cond1+'_'+cond2+'.png',bbox_inches='tight')


# #### distribution for absolute localization errors

# In[15]:


fig, ax = plt.subplots(2,2,figsize=(12,6))
cond = 1
if cond == 0:
    condition = "orientation"
    cond1 = 'Upright'
    cond2 = 'Inverted'
elif cond == 1:
    condition = "gender_acc"
    cond1 = 'Hit'
    cond2 = 'Miss'
else:
    condition = "holistic"
    cond1 = 'High-holistic'
    cond2 = 'Low-holistic'
subjIter = 0
for subj in subjs:
    sns.violinplot(y=np.abs(errAll[errAll['subj']==subj]['ang_error']),                x = errAll[errAll['subj']==subj][condition],                ax=ax[int(subjIter/2)][int(np.mod(subjIter,2))])
    ax[int(subjIter/2)][int(np.mod(subjIter,2))].set_xticklabels(labels=[cond2,cond1])
    ax[int(subjIter/2)][int(np.mod(subjIter,2))].set_xlabel('')
    subjIter += 1
ax[int(len(subjs)/2-1)][0].set_xlabel('Absolute localization errors (deg)')
plt.suptitle('Distribution of Absolute Localization Errors for '+cond1+' vs. '+cond2+' Faces')
plt.savefig('distribution_abs_err_'+cond1+'_'+cond2+'.png',bbox_inches='tight')


# #### distribution for localization errors variance

# In[72]:


errVar = errAll.groupby(['subj',condition,'locs'],as_index=False).agg({'ang_error':np.std})
fig, ax = plt.subplots(2,2,figsize=(12,6))
subjIter = 0
for subj in subjs:
    sns.kdeplot(x = errVar[errVar['subj']==subj]['ang_error'],                hue = errVar[errVar['subj']==subj][condition],                ax=ax[int(subjIter/2)][int(np.mod(subjIter,2))])
    ax[int(subjIter/2)][int(np.mod(subjIter,2))].legend(labels=[cond2,cond1])
    ax[int(subjIter/2)][int(np.mod(subjIter,2))].set_xlabel('')
    subjIter += 1
ax[int(subjIter/2-1)][0].set_xlabel('Variance of localization errors (deg)')
plt.suptitle('Distribution of Localization Errors Variance for '+cond1+' vs. '+cond2+' Faces')
plt.savefig('dist_var_err_'+cond1+'_'+cond2+'.png',bbox_inches='tight')


# ### Visualization

# Localization biases

# In[10]:


for subj in subjs:
    df1 = errAll[np.array(errAll['subj']==subj)&np.array(errAll['orientation']==0)]    .sort_values(['locs']).groupby('locs',as_index=False).agg(np.mean)
    df2 = errAll[np.array(errAll['subj']==subj)&np.array(errAll['orientation']==1)]    .sort_values(['locs']).groupby('locs',as_index=False).agg(np.mean)
    plt.figure()
    plt.plot(df1['locs'],df1['ang_error'],c='red',label = 'inverted')
    plt.plot(df2['locs'],df2['ang_error'],c='blue',label = 'upright')
    plt.legend()
    plt.title(subj)
    plt.savefig(subj+'.jpg')


# Localization precision

# In[23]:


errVar = errAll.groupby(['subj',condition,'locs'],as_index=False).agg({'ang_error':np.std})
for subj in subjs:
    df1 = errVar[np.array(errVar['subj']==subj)&np.array(errVar['orientation']==0)]    .sort_values(['locs']).groupby('locs',as_index=False).agg(np.mean)
    df2 = errVar[np.array(errVar['subj']==subj)&np.array(errVar['orientation']==1)]    .sort_values(['locs']).groupby('locs',as_index=False).agg(np.mean)
    plt.figure()
    plt.plot(df1['locs'],df1['ang_error'],c='red',label = 'inverted')
    plt.plot(df2['locs'],df2['ang_error'],c='blue',label = 'upright')
    plt.legend()
    plt.title(subj)
    plt.savefig(subj+'_locVar.jpg')


# # Within-subject correlations

# ## Bootstrap Within

# In[9]:


def SplitHalf(errAll,condition,val,coi,std=False,perm=False,allSubj = subjs):
    allRVals = list()
    for subj in allSubj:
        twoHalves = splitData(errAll[np.array(errAll[condition]==val)&np.array(errAll['subj']==subj)],coi,std)
        half1 = np.array(twoHalves.half1)
        if perm:
            half2 = np.array(shiftHalf(np.array(twoHalves.half2)))
        else:
            half2 = np.array(twoHalves.half2)
        [r,p]=pr(half1,half2)
        allRVals.append(r)
    allRVals = np.array(allRVals)
    return np.tanh(np.mean(np.arctanh(allRVals)))

def BtwCondWithin(data,condition,coi,std=False,perm=False,allSubj = subjs):
    allRVals = list()
    for subj in allSubj:
        twoHalves_cond1 = splitData(data[np.array(data[condition]==1)&np.array(data['subj']==subj)],coi,std)
        twoHalves_cond2 = splitData(data[np.array(data[condition]==0)&np.array(data['subj']==subj)],coi,std)
        half1 = np.array(twoHalves_cond1.half1)
        if perm:
            half2 = np.array(shiftHalf(np.array(twoHalves_cond2.half1)))
        else:
            half2 = np.array(twoHalves_cond2.half1)
        [r,p]=pr(half1,half2)
        allRVals.append(r)
    allRVals = np.array(allRVals)
    return np.tanh(np.mean(np.arctanh(allRVals)))


# ### Localization Biases

# In[10]:


bootWithinR = dict()
cond = 1
if cond == 0:
    condition = "orientation"
    cond1 = 'upright'
    cond2 = 'inverted'
elif cond == 1:
    condition = "gender_acc"
    cond1 = 'hit'
    cond2 = 'miss'
elif cond == 2:
    condition = "holistic"
    cond1 = 'high'
    cond2 = 'low'
elif cond == 3:
    condition == 'gender'
    cond1 = 'male'
    cond2 = 'female'
coi = 'ang_error'
bootWithinR[cond1] = list()
bootWithinR[cond2] = list()
bootWithinR['btwCond'] = list()
iters = 1000
data = errAll
for it in range(iters):
    bootWithinR[cond1].append(SplitHalf(data,condition,1,coi))
    bootWithinR[cond2].append(SplitHalf(data,condition,0,coi))
    bootWithinR['btwCond'].append(BtwCondWithin(data,condition,coi))  


# In[11]:


meanBootWithinR = [np.tanh(np.mean(np.arctanh(bootWithinR[cond1]))),                   np.tanh(np.mean(np.arctanh(bootWithinR[cond2]))),                  np.tanh(np.mean(np.arctanh(bootWithinR['btwCond'])))]
bootWithinCI = [[np.sort(bootWithinR[cond1])[np.int(iters*.025-1)],                 np.sort(bootWithinR[cond1])[np.int(iters*.975-1)]],               [np.sort(bootWithinR[cond2])[np.int(iters*.025-1)],                np.sort(bootWithinR[cond2])[np.int(iters*.975-1)]],               [np.sort(bootWithinR['btwCond'])[np.int(iters*.025-1)],                np.sort(bootWithinR['btwCond'])[np.int(iters*.975-1)]]]
print('Bootstrap Within Results')
print(cond1,': mean r =',meanBootWithinR[0],'95% CI =','[',bootWithinCI[0][0],bootWithinCI[0][1],']')
print(cond2,': mean r =',meanBootWithinR[1],'95% CI =','[',bootWithinCI[1][0],bootWithinCI[1][1],']')
print('Between',cond1,'and',cond2,': mean r =',meanBootWithinR[2],'95% CI =','[',bootWithinCI[2][0],bootWithinCI[2][1],']')


# In[12]:


two_tailed_p = 2*sum((np.array(bootWithinR[cond1])-                      np.array(bootWithinR['btwCond']))<0)/1000
two_tailed_p


# ### Localization Precision

# In[25]:


bootWithinPrecR = dict()
cond = 0
if cond == 0:
    condition = "orientation"
    cond1 = 'upright'
    cond2 = 'inverted'
elif cond == 1:
    condition = "gender_acc"
    cond1 = 'hit'
    cond2 = 'miss'
coi = 'ang_error'
bootWithinPrecR[cond1] = list()
bootWithinPrecR[cond2] = list()
bootWithinPrecR['btwCond'] = list()
iters = 1000
data = errAll
# data = errAll[errAll['orientation']==1]
for it in range(iters):
    bootWithinPrecR[cond1].append(SplitHalf(data,condition,1,coi,std=True))
    bootWithinPrecR[cond2].append(SplitHalf(data,condition,0,coi,std=True))
    bootWithinPrecR['btwCond'].append(BtwCondWithin(data,condition,coi,std=True))  


# In[26]:


meanBootWithinPrecR = [np.tanh(np.mean(np.arctanh(bootWithinPrecR[cond1]))),                   np.tanh(np.mean(np.arctanh(bootWithinPrecR[cond2]))),                  np.tanh(np.mean(np.arctanh(bootWithinPrecR['btwCond'])))]
bootWithinPrecCI = [[np.sort(bootWithinPrecR[cond1])[np.int(iters*.025-1)],                 np.sort(bootWithinPrecR[cond1])[np.int(iters*.975-1)]],               [np.sort(bootWithinPrecR[cond2])[np.int(iters*.025-1)],                np.sort(bootWithinPrecR[cond2])[np.int(iters*.975-1)]],               [np.sort(bootWithinPrecR['btwCond'])[np.int(iters*.025-1)],                np.sort(bootWithinPrecR['btwCond'])[np.int(iters*.975-1)]]]
print('Bootstrap Within Results')
print(cond1,': mean r =',meanBootWithinPrecR[0],'95% CI =','[',bootWithinPrecCI[0][0],bootWithinPrecCI[0][1],']')
print(cond2,': mean r =',meanBootWithinPrecR[1],'95% CI =','[',bootWithinPrecCI[1][0],bootWithinPrecCI[1][1],']')
print('Between',cond1,'and',cond2,': mean r =',meanBootWithinPrecR[2],'95% CI =','[',bootWithinPrecCI[2][0],bootWithinPrecCI[2][1],']')


# ## Permutation Within

# ### Permute Locations

# In[12]:


permWithinR = dict()
permWithinR['upright'] = list()
permWithinR['inverted'] = list()
permWithinR['btwCond'] = list()
iters = 1000
condition = 'orientation'
coi = 'ang_error'
data = errAll[errAll['orientation']==1]
for it in range(iters):
    permWithinR['upright'].append(avgSplitHalf(data,condition,1,coi,reps=16,perm=True))
    permWithinR['inverted'].append(avgSplitHalf(data,condition,0,coi,reps=16,perm=True))
    permWithinR['btwCond'].append(avgBtwCondWithin(data,condition,coi,reps=16,perm=True))


# In[153]:


meanPermWithinR = [np.tanh(np.mean(np.arctanh(permWithinR['upright']))),                   np.tanh(np.mean(np.arctanh(permWithinR['inverted']))),                   np.tanh(np.mean(np.arctanh(permWithinR['btwCond'])))]
permWithinCI = [[np.sort(permWithinR['upright'])[np.int(iters*.025-1)],                 np.sort(permWithinR['upright'])[np.int(iters*.975-1)]],               [np.sort(permWithinR['inverted'])[np.int(iters*.025-1)],                 np.sort(permWithinR['inverted'])[np.int(iters*.975-1)]],               [np.sort(permWithinR['btwCond'])[np.int(iters*.025-1)],                 np.sort(permWithinR['btwCond'])[np.int(iters*.975-1)]]]
print('Permutation Within Results')
print('Upright: mean r =',meanPermWithinR[0],'95% CI =','[',permWithinCI[0][0],permWithinCI[0][1],']')
print('Inverted: mean r =',meanPermWithinR[1],'95% CI =','[',permWithinCI[1][0],permWithinCI[1][1],']')
print('Between Upright and Inverted: mean r =',meanPermWithinR[2],'95% CI =','[',permWithinCI[2][0],permWithinCI[2][1],']')


# ### Permute Conditions

# In[215]:


def shuffle_cond_labels(errAll,cond,reps=32,subjName = subjs):
    df = errAll.copy()
    df = df.sort_values(['locs','subj']).reset_index(drop=True)
    df[cond] = np.array(df[cond].iloc[genRandInd(reps,False,len(subjName))])
    return df    


# In[216]:


permCondWithinR = dict()
permCondWithinR['upright'] = list()
permCondWithinR['inverted'] = list()
permCondWithinR['btwCond'] = list()
iters = 1000
errAll_shuffled = shuffle_cond_labels(errAll,condition)
for it in range(iters):
    permCondWithinR['upright'].append(avgSplitHalf(errAll_shuffled,condition,1,coi,reps=16))
    permCondWithinR['inverted'].append(avgSplitHalf(errAll_shuffled,condition,0,coi,reps=16))
    permCondWithinR['btwCond'].append(avgBtwCondWithin(errAll_shuffled,condition,coi,reps=16))


# In[217]:


meanPermCondWithinR = [np.tanh(np.mean(np.arctanh(permCondWithinR['upright']))),                   np.tanh(np.mean(np.arctanh(permCondWithinR['inverted']))),                   np.tanh(np.mean(np.arctanh(permCondWithinR['btwCond'])))]
permCondWithinCI = [[np.sort(permCondWithinR['upright'])[np.int(iters*.025-1)],                 np.sort(permCondWithinR['upright'])[np.int(iters*.975-1)]],               [np.sort(permCondWithinR['inverted'])[np.int(iters*.025-1)],                 np.sort(permCondWithinR['inverted'])[np.int(iters*.975-1)]],               [np.sort(permCondWithinR['btwCond'])[np.int(iters*.025-1)],                 np.sort(permCondWithinR['btwCond'])[np.int(iters*.975-1)]]]
print('Permutation Condition Label Results')
print('Upright: mean r =',meanPermCondWithinR[0],'95% CI =','[',permCondWithinCI[0][0],permCondWithinCI[0][1],']')
print('Inverted: mean r =',meanPermCondWithinR[1],'95% CI =','[',permCondWithinCI[1][0],permCondWithinCI[1][1],']')
print('Between Upright and Inverted: mean r =',meanPermCondWithinR[2],'95% CI =','[',permCondWithinCI[2][0],permCondWithinCI[2][1],']')


# # Between-subject correlations

# ## functions

# In[15]:


# correlate 2 halves from 2 subjects
def corrEachPair(errs1,errs2,coi,std=False,permOrNot=False):
    twoHalves1 = splitData(errs1,coi,std)
    twoHalves2 = splitData(errs2,coi,std)
    anotherHalf = twoHalves2.half1
    if permOrNot:
        anotherHalf = pd.Series(shiftHalf(np.array(anotherHalf)))
    [r,p]=pr(twoHalves1.half1,anotherHalf)        
    return r

# all pairwise between
def pairwiseBtw(errAll,condition,val,coi,std=False,subjName=subjs,perm=False):
    allRVals = list()
    allPairs = list(itertools.combinations(subjName,2))
    for pair in allPairs:
        r=corrEachPair(errAll[np.array(errAll['subj']==pair[0])&np.array(errAll[condition]==val)],                       errAll[np.array(errAll['subj']==pair[1])&np.array(errAll[condition]==val)],                       coi,std,perm)
        allRVals.append(r)
    return(np.array(allRVals))

# all pair-wise between condition between subject
def pairwiseBtwCond(data,condition,coi,std = False,subjName=subjs,perm=False):
    allRVals = list()
    allPairs = list(itertools.combinations(subjName,2))
    for pair in allPairs:           
        r=corrEachPair(errAll[np.array(errAll['subj']==pair[0])&np.array(errAll[condition]==0)],                       errAll[np.array(errAll['subj']==pair[1])&np.array(errAll[condition]==1)],                       coi,std,perm)
        allRVals.append(r)
    return(np.array(allRVals))


# ## Bootstrap Between

# ### Localization Biases

# In[16]:


cond = 1
if cond == 0:
    condition = "orientation"
    cond1 = 'upright'
    cond2 = 'inverted'
elif cond == 1:
    condition = "gender_acc"
    cond1 = 'hit'
    cond2 = 'miss'
elif cond == 2:
    condition = "holistic"
    cond1 = 'high'
    cond2 = 'low'
elif cond == 3:
    condition == 'gender'
    cond1 = 'male'
    cond2 = 'female'
coi = 'ang_error'
# data = errAll[errAll['orientation']==1]
data = errAll


# In[ ]:


iters = 1000
bootBtwR = dict()
bootBtwR[cond1] = list()
bootBtwR[cond2] = list()
bootBtwR['btwCond'] = list()
for it in range(iters):     
    bootBtwR[cond1].append(np.tanh(np.mean(np.arctanh(pairwiseBtw(data,condition,1,coi)))))
    bootBtwR[cond2].append(np.tanh(np.mean(np.arctanh(pairwiseBtw(data,condition,0,coi)))))
    bootBtwR['btwCond'].append(np.tanh(np.mean(np.arctanh(pairwiseBtwCond(data,condition,coi)))))
    
bootBtwR[cond1] = np.array(bootBtwR[cond1])
bootBtwR[cond2] = np.array(bootBtwR[cond2])
bootBtwR['btwCond'] = np.array(bootBtwR['btwCond'])


# In[ ]:


meanBootBtwR = [np.tanh(np.mean(np.arctanh(bootBtwR[cond1]))),               np.tanh(np.mean(np.arctanh(bootBtwR[cond2]))),               np.tanh(np.mean(np.arctanh(bootBtwR['btwCond'])))]
bootBtwCI = [[np.sort(bootBtwR[cond1])[np.int(iters*0.025-1)],              np.sort(bootBtwR[cond1])[np.int(iters*0.975-1)]],            [np.sort(bootBtwR[cond2])[np.int(iters*0.025-1)],              np.sort(bootBtwR[cond2])[np.int(iters*0.975-1)]],            [np.sort(bootBtwR['btwCond'])[np.int(iters*0.025-1)],              np.sort(bootBtwR['btwCond'])[np.int(iters*0.975-1)]]]
print('Bootstrap Between-subject Results')
print(cond1,': mean r =',meanBootBtwR[0],'95% CI =','[',bootBtwCI[0][0],bootBtwCI[0][1],']')
print(cond2,': mean r =',meanBootBtwR[1],'95% CI =','[',bootBtwCI[1][0],bootBtwCI[1][1],']')
print('Between',cond1,'and',cond2,': mean r =',meanBootBtwR[2],'95% CI =','[',bootBtwCI[2][0],bootBtwCI[2][1],']')


# ### Localization Precision

# In[27]:


bootPrecBtwR = dict()
bootPrecBtwR[cond1] = list()
bootPrecBtwR[cond2] = list()
bootPrecBtwR['btwCond'] = list()
for it in range(iters):     
    bootPrecBtwR[cond1].append(np.tanh(np.mean(np.arctanh(pairwiseBtw(data,condition,1,coi,True)))))
    bootPrecBtwR[cond2].append(np.tanh(np.mean(np.arctanh(pairwiseBtw(data,condition,0,coi,True)))))
    bootPrecBtwR['btwCond'].append(np.tanh(np.mean(np.arctanh(pairwiseBtwCond(data,condition,coi,True)))))
    
bootPrecBtwR[cond1] = np.array(bootPrecBtwR[cond1])
bootPrecBtwR[cond2] = np.array(bootPrecBtwR[cond2])
bootPrecBtwR['btwCond'] = np.array(bootPrecBtwR['btwCond'])


# In[28]:


meanBootPrecBtwR = [np.tanh(np.mean(np.arctanh(bootPrecBtwR[cond1]))),               np.tanh(np.mean(np.arctanh(bootPrecBtwR[cond2]))),               np.tanh(np.mean(np.arctanh(bootPrecBtwR['btwCond'])))]
bootPrecBtwCI = [[np.sort(bootPrecBtwR[cond1])[np.int(iters*0.025-1)],              np.sort(bootPrecBtwR[cond1])[np.int(iters*0.975-1)]],            [np.sort(bootPrecBtwR[cond2])[np.int(iters*0.025-1)],              np.sort(bootPrecBtwR[cond2])[np.int(iters*0.975-1)]],            [np.sort(bootPrecBtwR['btwCond'])[np.int(iters*0.025-1)],              np.sort(bootPrecBtwR['btwCond'])[np.int(iters*0.975-1)]]]
print('Bootstrap Between-subject Results')
print(cond1,': mean r =',meanBootPrecBtwR[0],'95% CI =','[',bootPrecBtwCI[0][0],bootPrecBtwCI[0][1],']')
print(cond2,': mean r =',meanBootPrecBtwR[1],'95% CI =','[',bootPrecBtwCI[1][0],bootPrecBtwCI[1][1],']')
print('Between',cond1,'and',cond2,': mean r =',meanBootPrecBtwR[2],'95% CI =','[',bootPrecBtwCI[2][0],bootPrecBtwCI[2][1],']')


# ## Permutation Between

# In[169]:


iters = 1000
permBtwR = dict()
permBtwR['upright'] = list()
permBtwR['inverted'] = list()
permBtwR['btwCond'] = list()
for it in range(iters):      
    permBtwR['upright'].append(np.tanh(np.mean(np.arctanh(pairwiseBtw(errAll,condition,1,coi,reps=16,perm=True)))))
    permBtwR['inverted'].append(np.tanh(np.mean(np.arctanh(pairwiseBtw(errAll,condition,1,coi,reps=16,perm=True)))))
    permBtwR['btwCond'].append(np.tanh(np.mean(np.arctanh(pairwiseBtwCond(errAll,condition,coi,reps=16,perm=True)))))
    
permBtwR['upright'] = np.array(permBtwR['upright'])
permBtwR['inverted'] = np.array(permBtwR['inverted'])
permBtwR['btwCond'] = np.array(permBtwR['btwCond'])


# In[170]:


meanPermBtwR = [np.tanh(np.mean(np.arctanh(permBtwR['upright']))),               np.tanh(np.mean(np.arctanh(permBtwR['inverted']))),               np.tanh(np.mean(np.arctanh(permBtwR['btwCond'])))]
permBtwCI = [[np.sort(permBtwR['upright'])[np.int(iters*0.025-1)],              np.sort(permBtwR['upright'])[np.int(iters*0.975-1)]],            [np.sort(permBtwR['inverted'])[np.int(iters*0.025-1)],              np.sort(permBtwR['inverted'])[np.int(iters*0.975-1)]],            [np.sort(permBtwR['btwCond'])[np.int(iters*0.025-1)],              np.sort(permBtwR['btwCond'])[np.int(iters*0.975-1)]]]
print('Permutation Between-subject Results')
print('Upright: mean r =',meanPermBtwR[0],'95% CI =','[',permBtwCI[0][0],permBtwCI[0][1],']')
print('Inverted: mean r =',meanPermBtwR[1],'95% CI =','[',permBtwCI[1][0],permBtwCI[1][1],']')
print('Between Upright and Inverted: mean r =',meanPermBtwR[2],'95% CI =','[',permBtwCI[2][0],permBtwCI[2][1],']')


# ## Ploting Within vs. Between

# In[30]:


plt.style.use('ggplot')
precisionAna = 1
x = ['Within-subject','Between-subject']
cond = [cond1,cond2,'Between Condition']
meanRs = [[meanBootWithinR[0],meanBootBtwR[0]],          [meanBootWithinR[1],meanBootBtwR[1]],          [meanBootWithinR[2],meanBootBtwR[2]]]
CIs = [np.array([[meanBootWithinR[0]-bootWithinCI[0][0],meanBootBtwR[0]-bootBtwCI[0][0]],                [bootWithinCI[0][1]-meanBootWithinR[0],bootBtwCI[0][1]-meanBootBtwR[0]]]),      np.array([[meanBootWithinR[1]-bootWithinCI[1][0],meanBootBtwR[1]-bootBtwCI[1][0]],                [bootWithinCI[1][1]-meanBootWithinR[1],bootBtwCI[1][1]-meanBootBtwR[1]]]),      np.array([[meanBootWithinR[2]-bootWithinCI[2][0],meanBootBtwR[2]-bootBtwCI[2][0]],                [bootWithinCI[2][1]-meanBootWithinR[2],bootBtwCI[2][1]-meanBootBtwR[2]]])]
if precisionAna == 1:
    meanRs = [[meanBootWithinPrecR[0],meanBootPrecBtwR[0]],          [meanBootWithinPrecR[1],meanBootPrecBtwR[1]],          [meanBootWithinPrecR[2],meanBootPrecBtwR[2]]]
    CIs = [np.array([[meanBootWithinPrecR[0]-bootWithinPrecCI[0][0],meanBootPrecBtwR[0]-bootPrecBtwCI[0][0]],                    [bootWithinPrecCI[0][1]-meanBootWithinPrecR[0],bootPrecBtwCI[0][1]-meanBootPrecBtwR[0]]]),          np.array([[meanBootWithinPrecR[1]-bootWithinPrecCI[1][0],meanBootPrecBtwR[1]-bootPrecBtwCI[1][0]],                    [bootWithinPrecCI[1][1]-meanBootWithinPrecR[1],bootPrecBtwCI[1][1]-meanBootPrecBtwR[1]]]),          np.array([[meanBootWithinPrecR[2]-bootWithinPrecCI[2][0],meanBootPrecBtwR[2]-bootPrecBtwCI[2][0]],                    [bootWithinPrecCI[2][1]-meanBootWithinPrecR[2],bootPrecBtwCI[2][1]-meanBootPrecBtwR[2]]])]
barWidth = .2
x_cond = [barWidth*0.5,barWidth*3.5,barWidth*6.5]
x_pos=[[0,barWidth],[barWidth*3,barWidth*4],[barWidth*6,barWidth*7]]
p1=plt.bar(x_pos[0], meanRs[0], color=['blue','yellow'],yerr=CIs[0],        width=barWidth,error_kw=dict(lw=2, capsize=5, capthick=3))
p2=plt.bar(x_pos[1], meanRs[1], color=['blue','yellow'],yerr=CIs[1],        width=barWidth,error_kw=dict(lw=2, capsize=5, capthick=3))         
p3=plt.bar(x_pos[2], meanRs[2], color=['blue','yellow'],yerr=CIs[2],        width=barWidth,error_kw=dict(lw=2, capsize=5, capthick=3))
# ii = 0
# for bar_pos in x_pos:    
#     plt.plot(np.array(bar_pos)-barWidth/2,\
#              [permWithinCI[ii][1],permWithinCI[ii][1]],\
#              color='black',linewidth=2)
#     plt.plot(np.array(bar_pos)+barWidth/2,\
#              [permBtwCI[ii][1],permBtwCI[ii][1]],\
#              color='black',linewidth=2)
#     ii += 1
plt.xticks(x_cond,cond,fontsize=14)
lgd=plt.legend(p1,x,loc=[1.1,0],fontsize=14)
# plt.errorbar([0,2*barWidth],[0,0],[[-permWithinCI[0],-permBtwCI[0]],\
#                                    [permWithinCI[1],permBtwCI[1]]],fmt='.k',\
#              capsize=40,ecolor='red')
plt.ylabel("Pearson's Correlation",fontsize=14)
ttl = plt.suptitle("Comparison of Within vs Between-subject Correlation in Different Conditions",fontsize=16)
if precisionAna == 1:
    plt.savefig('prec_'+cond1+cond2+"_WithinBetweenCorr_"+coi+".png",bbox_extra_artists=(lgd,ttl), bbox_inches='tight')
else:
    plt.savefig(cond1+cond2+"_WithinBetweenCorr_"+coi+".png",bbox_extra_artists=(lgd,ttl), bbox_inches='tight')
# # plot perm CIs as horizontal bars
# plt.plot([-barWidth/2,barWidth/2],[permWithinCI[0],permWithinCI[0]],color = 'red')
# plt.plot([-barWidth/2,barWidth/2],[permWithinCI[1],permWithinCI[1]],color = 'red')
# plt.plot([3*barWidth/2,5*barWidth/2],[permBtwCI[0],permBtwCI[0]],color = 'blue')
# plt.plot([3*barWidth/2,5*barWidth/2],[permBtwCI[1],permBtwCI[1]],color = 'blue')


# # Magnitude of Errors in Different Conditions

# ## functions

# In[57]:


def avgAcrossTrials(err,reps = 8,subjAll = subjs):
    avgErr = dict()
    for subj in subjAll:
        avgErr[subj] = err[subj].groupby('locs',as_index=False).errors.mean().errors
    return avgErr

def plotNbFace(face,nb,subjAll = subjs):
    fig, axs = plt.subplots(2, int(np.ceil(len(subjAll)/2)))
    itx = 0
    ity = 0
    num = 1
    for subj in subjAll:
        axs[itx,ity].scatter(nb[subj],face[subj])
        axs[itx,ity].plot(nb[subj],nb[subj],color='r')
        if num < np.ceil(len(subjAll)/2):
            ity += 1
            num += 1
        else:
            itx = 1
            ity = 0
            num = 1
    return None
       
def createDf(face,nb,DI = False,dataAll = dataAll,subjAll = subjs):
    df = pd.DataFrame()
    for subj in subjAll:
        toAdd = pd.DataFrame()
        if DI:
            toAdd['err'] = calSlope(face[subj],1).append(calSlope(nb[subj],1))
        else:
            toAdd['err'] = face[subj].append(nb[subj])
        toAdd['angles'] = np.vstack((np.arange(0,360,7.5).reshape(48,1),np.arange(0,360,7.5).reshape(48,1)))
        toAdd['subj'] = np.repeat(subj,96)
        toAdd['cond'] = np.vstack(((np.repeat('face',48)).reshape(48,1),(np.repeat('nb',48)).reshape(48,1)))
        toAdd['age'] = dataAll[subj].age
        toAdd['gender'] = dataAll[subj].gender
        toAdd['absErr'] = abs(toAdd['err'])
        df = df.append(toAdd)
    df.index = np.arange(df.shape[0])
    return df
    
def errDiff(face,nb,subjAll = subjs):
    df = pd.DataFrame()
    for subj in subjAll:
        toAdd = pd.DataFrame()
        toAdd['diff_val'] = avgNoise[subj] - avgFace[subj]
        toAdd['angles'] = np.arange(0,360,7.5)
        toAdd['subj'] = np.repeat(subj,48)
        df = df.append(toAdd)
    df.index = np.arange(df.shape[0])
    return df


# ## Construct DateFrame

# In[58]:


DI = False
if analysisInput == 2:
    face = rtFace
    nb = rtNoise
elif analysisInput == 4:
    face = eccFace
    nb = eccNoise
else:
    face = errFace
    nb = errNoise  
if analysisInput == 3:
    DI = True
avgFace = avgAcrossTrials(face)
avgNoise = avgAcrossTrials(nb)    
df = createDf(avgFace,avgNoise,DI)
df['LoVF'] = df.angles<180
df['LeVF'] = (df.angles>90)&(df.angles<=270)
df['Horz'] = (df.angles<30)|((df.angles>=150)&(df.angles<210))|(df.angles>=330)
df['Vert'] = ((df.angles>=60)&(df.angles<120))|((df.angles>=240)&(df.angles<300))
df['Oblique'] = (~df['Horz'])&(~df['Vert'])
display(df.head(),avgFace['EW'])


# In[151]:


df4 = pd.DataFrame()
for ss in subjs:
    tmp_df = df[df.subj==ss].pivot_table(index="angles",columns="cond",values='absErr',aggfunc=np.mean)
    tmp_df['subj'] = ss
    plt.figure()
    plt.plot(tmp_df.index,tmp_df.face,c='red',label='face')
    plt.plot(tmp_df.index,tmp_df.nb,c='blue',label='noise')
    plt.legend()
#     plt.savefig("absEccErrorAcrossAngles_"+ss+".png")
    df4 = pd.concat([df4,tmp_df])
# sns.kdeplot(x=tmp_df['face'],y=tmp_df['nb'])
plt.figure()
plt.hist(df4['face']-df4['nb'])
plt.savefig("absEccErrorDifference.png")
ttest(np.array(df4['face']),np.array(df4['nb']))


# In[51]:


plt.hist(df.loc[df.cond=='face','err'],alpha=0.5,label="Face")
plt.hist(df.loc[df.cond=='nb','err'],alpha=0.5,label="Noise Patch")
plt.legend()
plt.savefig('eccerrorDistribution.png')


# ## Modeling in R

# ### Compare different conditions and individual diffs

# In[59]:


get_ipython().run_cell_magic('R', '-i df', 'library(nlme)\nlibrary(car)\ndf$subj = factor(df$subj)\ndf$cond = factor(df$cond)\ndf$Horz = factor(df$Horz)\ndf$Vert = factor(df$Vert)\n# df$gender = factor(df$gender)\n# df$angles = factor(df$angles)\n# model_lme <- lme(err~cond+(LoVF*Vert+LoVF*LeVF+LeVF*Horz),random = ~1|subj,data=df)\nmodel_abs_lme <- lme(absErr~cond*(LoVF+LeVF),random = ~1|subj,data=df)\nmodel_raw_lme <- lme(err~cond*(LoVF+LeVF),random = ~1|subj,data=df)\n# model_lm <- lm(err~cond*(LoVF*Vert+LoVF*LeVF+LeVF*Horz),data=df)\n# model_lm <- lm(err~cond,data=df)\nmodel_null <- lm(err~1,data=df)\nsummary(model_raw_lme)\n# summary(model_abs_lme)\n\n# Anova(model_abs_lme)\n# Anova(model_raw_lme)\n# anova(model_lme,model_lm)')


# ### Visualize the comparisons

# In[32]:


def boot_ci(data,errType,ci=95,iters=1000):
    allMean = []
    realMean = np.mean(data[errType])
    for i in range(iters):
        randMean = []
        for subj in data.subj.unique():
            randMean.append(np.mean(np.random.choice(data.loc[data.subj==subj,errType],len(data.subj),replace=True)))
        allMean.append(np.mean(randMean))
    allMean = np.sort(np.array(allMean))
    return np.array([realMean-allMean[int(iters*((100-ci)/200))],allMean[int(iters*((100+ci)/200))]-realMean])


# In[34]:


# Dataframe for plotting
plotdf=df.copy()
plotdf.loc[(plotdf.LoVF)&(plotdf.LeVF),'quad'] = 'LoLe'
plotdf.loc[(plotdf.LoVF)&(~plotdf.LeVF),'quad'] = 'LoRi'
plotdf.loc[(~plotdf.LoVF)&(plotdf.LeVF),'quad'] = 'UpLe'
plotdf.loc[(~plotdf.LoVF)&(~plotdf.LeVF),'quad'] = 'UpRi'

# within-subject error bar estimation
# # first, center the data
# aggTable = plotdf.groupby(by=["cond","quad","subj"],as_index=False).err.mean()
# aggTableCopy = aggTable.copy()
# for subj in aggTable.subj.unique():
#     aggTable.loc[aggTable.subj == subj,"err"] = \
#     aggTable.loc[aggTable.subj == subj,"err"] - np.mean(aggTable.loc[aggTable.subj == subj,"err"])
# aggTable.loc[:,"err"] = aggTable.loc[:,"err"]+np.mean(aggTableCopy.err)
# # then, incorporate the correction factor
# aggTableCopy = aggTable.copy()
# cf = np.sqrt(2*2*2/2*2*2-1)
# for cond in aggTable.cond.unique():
#     for quad in aggTable.quad.unique():
#         condMean = np.mean(aggTable.loc[(aggTable.cond == cond)&(aggTable.quad == quad),"err"])
#         aggTable.loc[(aggTable.cond == cond)&(aggTable.quad == quad),"err"] = \
#         cf*(aggTable.loc[(aggTable.cond == cond)&(aggTable.quad == quad),"err"] - condMean) + condMean
# # finally, calculate the standard error
errBarRaw = pd.DataFrame()
errBarAbs = pd.DataFrame()
for cond in  plotdf.cond.unique():
    for quad in plotdf.quad.unique():
        d = plotdf.loc[(plotdf.cond==cond)&(plotdf.quad==quad)]
        errBarRaw[cond+quad] = boot_ci(d,"err")
        errBarAbs[cond+quad] = boot_ci(d,"absErr")
# errBarAbs = plotdf.groupby(by=["cond","quad","subj"],as_index=False).absErr.mean().\
# groupby(by=["cond","quad"],as_index=False).agg({"absErr":np.std})
# errBarAbs["absErr"] = errBarAbs.absErr/np.sqrt(7)


# plot options
if analysisInput == 4:
    errType = "Eccentricity"
elif analysisInput == 1:
    errType = "Angular"
else:
    errType= "rt_"


# In[35]:


# plotdf.head()
# boot_ci(plotdf[plotdf.quad=="LoLe"],"err")
# errBarRaw.head()
errBarRaw = np.array(errBarRaw)
errBarAbs = np.array(errBarAbs)


# In[39]:


plt.style.use('ggplot')
# Raw Errors Plot
g = sns.catplot(
    data=plotdf, kind="bar",
    x="cond", y="err", hue="quad",
    palette="dark", alpha=.6, height=6,
    legend_out = True, ci=None
)
xl=plt.xlabel("Stimulus Type")
plt.xticks(ticks=[0,1],labels=["face","noise patch"])
if analysisInput == 2:
    plt.ylabel("Response Time (second)")
    ttl=plt.title("RT Across Hemifields with Different Stimuli")
else:
    plt.ylabel(errType+" Errors")
    ttl=plt.title(errType+" Errors Across Hemifields with Different Stimuli")
# title
new_title = 'Quadrants'
g._legend.set_title(new_title)
g._legend.set_bbox_to_anchor([1.1,0.9])
# replace labels
new_labels = ["Lower-Right","Lower-Left","Upper-Left","Upper-Right"]
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)

# plot customized error bars
x_coords = np.hstack((np.arange(-0.3,0.5,0.2),np.arange(0.7,1.5,0.2)))
y_coords = []
uniqQuad = plotdf.quad.unique()
uniqCond = plotdf.cond.unique()
for cond in uniqCond:
    for quad in uniqQuad:
        y_coords.append(np.mean(plotdf.loc[(plotdf.cond==cond)&(plotdf.quad==quad),"err"]))
plt.errorbar(x_coords,y_coords, yerr=errBarRaw,c="black",fmt=' ', zorder=-1)
# save figure
g.savefig(errType[:3]+"ErrorAcrossHemifields.png",bbox_extra_artists=(xl,ttl))


# In[26]:


# Absolute errors
# upper vs lower hemifield
g = sns.catplot(
    data=plotdf, kind="bar",
    x="cond", y="absErr", hue="quad",
    palette="dark", alpha=.6, height=6,
    legend_out = True, ci= None
)
xl=plt.xlabel("Stimulus Type")
plt.xticks(ticks=[0,1],labels=["face","noise patch"])

plt.ylabel("Absolute " + errType + " Errors")
ttl=plt.title("Absolute "+ errType + " Errors Across Hemifields with Different Stimuli")
# title
new_title = 'Quadrants'
g._legend.set_title(new_title)
g._legend.set_bbox_to_anchor([0.9,0.9])
# replace labels
new_labels = ["Lower-Right","Lower-Left","Upper-Left","Upper-Right"]
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
# plot customized error bars
x_coords = np.hstack((np.arange(-0.3,0.5,0.2),np.arange(0.7,1.5,0.2)))
y_coords = []
uniqQuad = plotdf.quad.unique()
uniqCond = plotdf.cond.unique()
for cond in uniqCond:
    for quad in uniqQuad:
        y_coords.append(np.mean(plotdf.loc[(plotdf.cond==cond)&(plotdf.quad==quad),"absErr"]))
plt.errorbar(x_coords,y_coords, yerr=errBarAbs,c="black",fmt=' ', zorder=-1)
# save fig
plt.savefig("abs"+errType[:3]+"ErrorAcrossHemifields.png",bbox_extra_artists=(xl,ttl), bbox_inches='tight')


# ## Plot in ggplot2

# In[85]:


get_ipython().run_cell_magic('R', '-i df', 'library(ggplot2)\nlibrary(gridExtra)\nsubjs = unique(df$subj)\nfor (ss in subjs) {\n    df_face = (df[df$subj == ss,])[0:48,]\n    df_nb = (df[df$subj == ss,])[48:96,]\n    # faces\n    p1 = ggplot(df_face, aes(x=angles,y=2, fill=err)) +\n    geom_tile() +\n    ylim(c(1,3)) +\n    theme(panel.background = element_rect(fill = \'white\'),\n         legend.position = \'none\') +\n    scale_colour_gradient2(low = "blue", mid = "white",midpoint=mean(0.55),\n                         high ="red",space = "Lab",\n                         guide = guide_colourbar(direction = "horizontal"), \n                         aesthetics = "fill",limits = c(0.47,0.62)) +  \n    coord_polar(start=pi/2, direction=1)\n    # noise blob\n    p2 = ggplot(df_nb, aes(x=angles,y=2, fill=err)) + \n    geom_tile() +\n    ylim(c(1,3)) +\n    theme(panel.background = element_rect(fill = \'white\'),\n         legend.position = \'none\') +\n    scale_colour_gradient2(low = "blue", mid = "white",midpoint=mean(0.55),\n                         high ="red",space = "Lab",\n                         guide = guide_colourbar(direction = "horizontal"), \n                         aesthetics = "fill",limits = c(0.47,0.62)) +  \n    coord_polar(start=pi/2, direction=1)\n    f <- arrangeGrob(p1, p2, nrow = 1)\n    # save fig\n    if (ss==\'EW\') {\n        ggsave(paste(ss,"_rt.png",sep=""),f)\n    }\n    \n}')


# In[171]:


display(df.groupby(by=['LoVF','LeVF']).err.mean(),       df.groupby(by=['Horz','LeVF']).err.mean(),       df.groupby(by=['Vert','LoVF']).err.mean())


# # Unique Variance Analysis

# ## Preparation

# In[445]:


DI = False
if analysisInput == 2:
    data5 = rtAll
elif analysisInput == 4:
    data5 = eccAll
else:
    data5 = errAll
if analysisInput == 3:
    DI = True

# df for analysis 5
df5 = pd.DataFrame()
for subjName in subjs:
    tmp = pd.DataFrame()
    self_nb = avgHalf(data5.loc[(data5.subj==subjName)&(data5.stim),:].                      sort_values('locs').reset_index(drop=True),genRandInd())
    self_face = avgHalf(data5.loc[(data5.subj==subjName)&(~(data5.stim)),:].                      sort_values('locs').reset_index(drop=True),genRandInd())
    other_nb = avgHalf(data5.loc[(~(data5.subj==subjName))&(data5.stim),:].                    sort_values('locs').reset_index(drop=True),genRandInd(subjNum=len(subjs)-1))
    other_face = avgHalf(data5.loc[(~(data5.subj==subjName))&(~(data5.stim)),:].                    sort_values('locs').reset_index(drop=True),genRandInd(subjNum=len(subjs)-1))
    if DI:
        tmp['err'] = calSlope(self_nb.half1)
        tmp['selfNB'] = calSlope(self_nb.half2)
        tmp['selfFace'] = calSlope(self_face.half1)
        tmp['otherNB'] = calSlope(other_nb.half1)
        tmp['otherFace'] = calSlope(other_face.half1)
    else:
        tmp['err'] = self_nb.half1
        tmp['selfNB'] = self_nb.half2
        tmp['selfFace'] = self_face.half1
        tmp['otherNB'] = other_nb.half1
        tmp['otherFace'] = other_face.half1
    tmp['subj'] = subjName
    df5 = pd.concat([df5,tmp])
df5=df5.reset_index(drop=True)


# In[446]:


df5.tail()


# ## Model

# In[448]:


get_ipython().run_cell_magic('R', '-i df5', 'library(nlme)\nlibrary(ggplot2)\n# first construct different models\nmdl5_full <- lm(err~selfNB+selfFace+otherNB+otherFace,data=df5)\nmdl5_noNB <- lm(err~selfFace+otherNB+otherFace,data=df5)\nmdl5_noFace <- lm(err~selfNB+otherNB+otherFace,data=df5)\nmdl5_noMeanNb <- lm(err~selfNB+selfFace+otherFace,data=df5)\nmdl5_noMeanFace <- lm(err~selfNB+selfFace+otherNB,data=df5)\nmdl5_null <- lm(err~1,data=df5)\n\n# now calculate the unique variance\nuniq_Nb <- (summary(mdl5_full)$r.squared - summary(mdl5_noNB)$r.squared)\nuniq_face <- (summary(mdl5_full)$r.squared - summary(mdl5_noFace)$r.squared)\nuniq_meanNb <- (summary(mdl5_full)$r.squared - summary(mdl5_noMeanNb)$r.squared)\nuniq_meanFace <- (summary(mdl5_full)$r.squared - summary(mdl5_noMeanFace)$r.squared)\nunexp <- 1 - summary(mdl5_full)$r.squared\nshared <- 1 - uniq_face - uniq_meanNb - uniq_meanFace - unexp\n# plot the unique variance percentage\ndata <- data.frame(\n  x=c("selfNb","selfFace","meanNb","meanFace","shared","unexplained") ,  \n  y=c(uniq_Nb,uniq_face,uniq_meanNb,uniq_meanFace,shared,unexp)\n  )\nggplot(data,aes(x=x, y=y))+\n geom_bar(stat = "identity")\n# print(uniq_face)\n# print(uniq_meanNb)\n# print(uniq_meanFace)\nsummary(mdl5_full)')


# # Serial Dependence

# ## Preparation

# In[249]:


# first decide which data to use; it does no make sense to analyze distortion indices for SD
if analysisInput == 1 or analysisInput == 3:
    data6 = errAll
elif analysisInput == 2:
    data6 = rtAll
else:
    data6 = eccAll

# Now construct the dataframe
df6 = pd.DataFrame()
df6_sc = pd.DataFrame()
for subj in subjs:
    tmp2 = pd.DataFrame()
    tmp3 = pd.DataFrame()
    tmp1 = data6.loc[data6.subj==subj,:].reset_index(drop=True)
    tmp1.loc[~tmp1.stim,'locs'] = tmp1.loc[~tmp1.stim,'locs'] - 48
    # n-1 trials
    tmp2['errors'] = np.array(tmp1.iloc[1:-1,0]) 
    tmp2['prevErr'] = np.array(tmp1.iloc[:-2,0])
    tmp2['prevLoc'] = np.array(tmp1.iloc[:-2,1])
    tmp2['currLoc'] = np.array(tmp1.iloc[1:-1,1])
    tmp2['prevStim'] = list(tmp1.iloc[:-2,2])
    tmp2['currStim'] = list(tmp1.iloc[1:-1,2])
    tmp2['rt'] = np.array(rtAll[rtAll.subj==subj].reset_index(drop=True).iloc[1:-1,0])
    tmp2['angdiff'] = 7.5*recenter(np.array(tmp1.iloc[:-2,1])-np.array(tmp1.iloc[1:-1,1])) #prev-curr
    tmp2['perceived_angdiff'] = 7.5*recenter(np.array(tmp1.iloc[:-2,1])     - np.array(tmp1.iloc[1:-1,1])) + tmp2['prevErr'] #prev perceived loc - curr loc
    tmp2['stimdiff'] = tmp2['prevStim']==tmp2['currStim']
    tmp2['subj'] = subj
    tmp2 = polyCorrection(tmp2)
    df6 = pd.concat([df6,tmp2])
    # n+1 trials, sanity check
    tmp3['errors'] = np.array(tmp1.iloc[1:-1,0]) 
    tmp3['futErr'] = np.array(tmp1.iloc[2:,0])
    tmp3['futLoc'] = np.array(tmp1.iloc[2:,1])
    tmp3['currLoc'] = np.array(tmp1.iloc[1:-1,1])
    tmp3['futStim'] = list(tmp1.iloc[2:,2])
    tmp3['currStim'] = list(tmp1.iloc[1:-1,2])
    tmp3['angdiff'] = recenter(np.array(tmp1.iloc[2:,1])-np.array(tmp1.iloc[1:-1,1]))#future-curr
    tmp3['subj'] = subj
    df6_sc = pd.concat([df6_sc,tmp3])
    # update index
    df6.index = range(df6.shape[0])
    df6_sc.index = range(df6_sc.shape[0])
df6 = polyCorrection(df6,outCol='correctedGroupError')


# In[250]:


df6.head()


# ## Utils for VonMise

# In[251]:


# Some global variables
bootA = []
permA = []
# functions
def vonmise_derivative(xdata, a, kai):
    xdata = xdata / 180 * np.pi
    return - a / (i0(kai) * 2 * np.pi) * np.exp(kai * np.cos(xdata)) * kai * np.sin(xdata) # Derivative of vonmise formula

def getRegressionLine(x, y, peak):
    stimuli_diff_filtered = []
    filtered_responseError_new = []
    for i in range(len(x)):
        if x[i] < peak + 1 and x[i] > - peak + 1:
            stimuli_diff_filtered.append(x[i])
            filtered_responseError_new.append(y[i])
    coef = np.polyfit(stimuli_diff_filtered,filtered_responseError_new,1)
    poly1d_fn = np.poly1d(coef)
    return poly1d_fn, coef

def polyFunc(x, coeffs):
    y = 0
    order = len(coeffs)
    for i in range(order):
        y += coeffs[i] * (x ** (order - 1 - i))
    return y

def polyCorrection(df,errCol='errors',outCol='correctedError'):
    coefs = np.polyfit(df['currLoc'], df[errCol],10) # polynomial coefs
    df[outCol] = [y - polyFunc(x, coefs) for x,y in zip(df['currLoc'], df[errCol])]
    temp_error = df[outCol].copy()
    df[outCol] = recenter(temp_error)
    return df

def CurvefitFunc(x, y, func=vonmise_derivative, init_vals=[-30, 3], bounds_input = ([-60,0.5],[60, 20])):
    best_vals, covar = curve_fit(func, x, y, p0=init_vals, bounds = bounds_input)
    return best_vals

def VonMise_fitting(x, y, x_range, boot=False,perm=False,func=vonmise_derivative, init_vals=[-30, 3],  bounds_input = ([-60,0.5],[60,20])):
    best_vals = CurvefitFunc(x, y, init_vals=init_vals, bounds_input = bounds_input)

    if boot:
        OutA = [] # Output a array, store each trial's a
        outSlope = []
        outIntercept = []
        bsSize = int(1.0 * len(x))
        bootiter=1000
        for i in range(bootiter):
            RandIndex = np.random.choice(len(x), bsSize, replace=True) # get randi index of xdata
            xdataNEW = [x[j] for j in RandIndex] # change xdata index
            ydataNEW = [y[j] for j in RandIndex] # change ydata index
            try:
                temp_best_vals = CurvefitFunc(xdataNEW, ydataNEW, init_vals=init_vals, bounds_input=bounds_input)
                new_x = np.linspace(-x_range, x_range, 300)
                new_y = [vonmise_derivative(xi,temp_best_vals[0],temp_best_vals[1]) for xi in new_x]
                if new_x[np.argmax(new_y)] > 0: 
                    OutA.append(np.max(new_y))
                else: 
                    OutA.append(-np.max(new_y))

#                 poly1d_fn, coef = getRegressionLine(xdataNEW, ydataNEW, self.peak_x)
#                 outSlope.append(coef[0])
#                 outIntercept.append(coef[1])
            except RuntimeError:
                pass
        print("bs_a:",round(np.mean(OutA),2),"	95% CI:",np.percentile(OutA,[2.5,97.5]))
#         bootA = OutA  ###ADD ME BACK YO
#         self.outSlope = outSlope
#         self.outIntercept = outIntercept
        # np.save(self.result_folder + 'bootstrap.npy', OutA)

    if perm:
        # perm_a, perm_b = repeate_sampling('perm', xdata, ydata, CurvefitFunc, size = permSize)
        OutB = [] # Output a array, store each trial's a
        perm_xdata = x
        permIter = 1000
        for i in range(permIter):
            perm_xdata = np.random.permutation(perm_xdata) # permutate nonlocal xdata to update, don't change ydata
            try:
                temp_best_vals = CurvefitFunc(perm_xdata, y, init_vals=init_vals, bounds_input=bounds_input) # permutation make a sample * range(size) times
                new_x = np.linspace(-x_range, x_range, 300)
                new_y = [vonmise_derivative(xi,temp_best_vals[0],temp_best_vals[1]) for xi in new_x]
                if new_x[np.argmax(new_y)] > 0: 
                    OutB.append(np.max(new_y))
                else: 
                    OutB.append(-np.max(new_y))
            except RuntimeError:
                pass
        print("perm_a:",round(np.mean(OutB),2),"	95% CI:",np.percentile(OutB,[5,95]))
        
    print('Von Mise Parameters: amplitude {0:.4f}, Kai {1:.4f}.'.format(best_vals[0],best_vals[1]))
    return best_vals#, outSlope, outIntercept
    #return OutA ###TAKE OUT OUTA YO
    


# In[271]:


best_vals_face = dict()
best_vals_nb = dict()
condi='prevStim'
val='angdiff'
for ss in subjs:
    best_vals_face[ss]=VonMise_fitting(np.array(df6.loc[(df6.subj==ss)&(~df6[condi]),val]),                                  np.array(df6[(df6.subj==ss)&(~df6[condi])]['correctedError']),180)
    best_vals_nb[ss]=VonMise_fitting(np.array(df6.loc[(df6.subj==ss)&(df6[condi]),val]),                                  np.array(df6[(df6.subj==ss)&(df6[condi])]['correctedError']),180)


best_vals_group_nb=VonMise_fitting(np.array(df6[df6[condi]][val]),                                   np.array(df6[df6[condi]]['correctedGroupError']),180)
best_vals_group_face=VonMise_fitting(np.array(df6[~df6[condi]][val]),                                     np.array(df6[~df6[condi]]['correctedGroupError']),180)


# ## Visualize Raw Data

# In[273]:


# display(df.head(),data6['EW'].head()
# display(df.head(),df_sc.head(),data6['EW'].head(),data6['EW'].iloc[1:-1,0])
df6['rounded'] = df6['perceived_angdiff'].round()
condi = 'prevStim'
val='angdiff'
amp_nb=list()
amp_face=list()
for ss in subjs:
    plt.figure()
    ax=plt.subplot(1,2,1)
    plt.scatter(x=df6.loc[(df6.subj==ss)&(df6[condi]),val],            y=df6.loc[(df6.subj==ss)&(df6[condi]),'correctedError'])
    running_mean = df6.loc[(df6.subj==ss)&(df6[condi]),:].groupby(by=val).correctedError.mean()
    plt.plot(running_mean,color='red')
    # plot von mises fit
    x_range = 180
    new_x = np.linspace(-x_range, x_range, 360)
    new_y = [vonmise_derivative(xi,best_vals_nb[ss][0],best_vals_nb[ss][1]) for xi in new_x]
    if new_x[np.argmax(new_y)] > 0: 
        amp_nb.append(np.max(new_y))
    else: 
        amp_nb.append(-np.max(new_y))
    plt.plot(new_x, new_y, 'k-', linewidth = 4)
    ax.set_title("Previous Stimuli Are Noise Patches")
    
    
    ax=plt.subplot(1,2,2)
    plt.scatter(x=df6.loc[(df6.subj==ss)&(~df6[condi]),val],            y=df6.loc[(df6.subj==ss)&(~df6[condi]),'correctedError'])
#     plt.xlim([-5,5])
#     plt.ylim([-10,10])
    running_mean = df6.loc[(df6.subj==ss)&(~df6[condi]),:].groupby(by=val).correctedError.mean()
    plt.plot(running_mean,color='red')
    # plot von mises fit
    x_range = 180
    new_x = np.linspace(-x_range, x_range, 360)
    new_y = [vonmise_derivative(xi,best_vals_face[ss][0],best_vals_face[ss][1]) for xi in new_x]
    second_x = np.linspace(-x_range, x_range, 360)
    DoVM_values = [vonmise_derivative(xi,best_vals_face[ss][0],best_vals_face[ss][1]) for xi in second_x]
    plt.plot(new_x, new_y, 'k-', linewidth = 4)
    if new_x[np.argmax(new_y)] > 0: 
        amp_face.append(np.max(new_y))
    else: 
        amp_face.append(-np.max(new_y))
    ax.set_title("Previous Stimuli Are Faces")
    plt.savefig(ss+"_prevStim"+".png")
#     plt.scatter(x=df.loc[(df.subj==ss)&(df.prevStim==df.currStim),'prevErr'],\
#                 y=df.loc[(df.subj==ss)&(df.prevStim==df.currStim),'errors'])
#     plt.scatter(x=df.loc[(df.subj==ss)&(~df.prevStim==df.currStim),'prevErr'],\
#                 y=df.loc[(df.subj==ss)&(~df.prevStim==df.currStim),'errors'],c='green')
#     running_mean_same = df.loc[(df.subj==ss)&(df.prevStim==df.currStim),:].groupby(by='prevErr').errors.mean()
#     running_mean_diff = df.loc[(df.subj==ss)&(~df.prevStim==df.currStim),:].groupby(by='prevErr').errors.mean()
#     plt.plot(running_mean_same,color='red')
#     plt.plot(running_mean_diff,color='orange')

# Plot Group Result
plt.figure()
plt.scatter(x=df6.loc[df6[condi],val],            y=df6.loc[df6[condi],'correctedGroupError'])
running_mean = df6[df6[condi]].groupby(by=val).correctedGroupError.mean()
plt.plot(running_mean,color='red')
x_range = 180
new_x = np.linspace(-x_range, x_range, 360)
new_y = [vonmise_derivative(xi,best_vals_group_nb[0],best_vals_group_nb[1]) for xi in new_x]
second_x = np.linspace(-x_range, x_range, 360)
DoVM_values = [vonmise_derivative(xi,best_vals_group_nb[0],best_vals_group_nb[1]) for xi in second_x]
plt.plot(new_x, new_y, 'k-', linewidth = 4)
if new_x[np.argmax(new_y)] > 0:
    plt.title("half amplitude = {0:.4f}, half width = {1:.4f}". format(np.max(new_y), new_x[np.argmax(new_y)]))
else: 
    plt.title("half amplitude = {0:.4f}, half width = {1:.4f}". format(-np.max(new_y), -new_x[np.argmax(new_y)]))
plt.savefig("group_prevStim_nb.png")
# Plot Group Result - PrevStim Face
plt.figure()
plt.scatter(x=df6.loc[~df6[condi],val],            y=df6.loc[~df6[condi],'correctedGroupError'])
running_mean = df6[~df6[condi]].groupby(by=val).correctedGroupError.mean()
plt.plot(running_mean,color='red')
x_range = 180
new_x = np.linspace(-x_range, x_range, 360)
new_y = [vonmise_derivative(xi,best_vals_group_face[0],best_vals_group_face[1]) for xi in new_x]
second_x = np.linspace(-x_range, x_range, 360)
DoVM_values = [vonmise_derivative(xi,best_vals_group_face[0],best_vals_group_face[1]) for xi in second_x]
plt.plot(new_x, new_y, 'k-', linewidth = 4)
if new_x[np.argmax(new_y)] > 0:
    plt.title("half amplitude = {0:.4f}, half width = {1:.4f}". format(np.max(new_y), new_x[np.argmax(new_y)]))
else: 
    plt.title("half amplitude = {0:.4f}, half width = {1:.4f}". format(-np.max(new_y), -new_x[np.argmax(new_y)]))
plt.savefig("group_prevStim_face.png")
# plt.scatter(x=df.loc[(~df.prevStim),'prevErr'],\
#             y=df.loc[(~df.prevStim),'errors'],c='green')
# running_mean_same = df.loc[(df.prevStim==df.currStim),:].groupby(by='prevErr').errors.mean()
# running_mean_diff = df.loc[(~df.prevStim==df.currStim),:].groupby(by='prevErr').errors.mean()
# # plt.plot(running_mean_same,color='red')
# # plt.plot(running_mean_diff,color='orange')
# plt.figure()
# plt.plot(x=np.array(circ_moving_average(running_mean_diff.index,circ=False)),\
#          y=np.array(circ_moving_average(running_mean_diff,circ=False)))
# plt.xlim([-20,20])
# plt.ylim([-20,20])
# plt.xlim([-5,5])
# plt.ylim([-20,20])


# In[275]:


plt.scatter(x=amp_nb,y=amp_face)


# In[224]:


plt.scatter(df6[~df6.prevStim]['angdiff'],df6[~df6.prevStim]['perceived_angdiff'])
plt.plot(np.arange(-180,180),np.arange(-180,180),c='red')


# ## Modeling in R

# In[246]:


get_ipython().run_cell_magic('R', '-i df6', 'library(nlme)\n# df$angles = factor(df$angles)\nmodel_lm <- lm(correctedError~(angdiff+perceived_angdiff)*(prevStim*stimdiff),data=df6)\nprint("Full Model")\nprint(summary(model_lm))\ndataPrevNoise = subset(df6, prevStim == TRUE)\ndataPrevFace = subset(df6, prevStim == FALSE)\nmdl_noise <- lm(correctedError~perceived_angdiff+angdiff,data=dataPrevNoise)\nmdl_face <- lm(correctedError~perceived_angdiff+angdiff,data=dataPrevFace)\nprint("Model: Previous Stimuli Are NP")\nprint(summary(mdl_noise))\nprint("Model: Previous Stimuli Are FACES")\nprint(summary(mdl_face))')


# Noise: prevErr     b = -0.04, p = .059 <br>
# Face:  prevErr     b = 0.08,  p < .001 <br>

# In[212]:


get_ipython().run_cell_magic('R', '-i df_sc', 'model_lm <- lm(errors~(futErr+angdiff)*futStim,data=df_sc)\n# dataPrevNoise = subset(df_sc, prevStim == TRUE)\n# dataPrevFace = subset(df_sc, prevStim == FALSE)\n# mdl_noise <- lm(errors~prevErr+angdiff,data=dataPrevNoise)\n# mdl_face <- lm(errors~prevErr+angdiff,data=dataPrevFace)\nsummary(model_lm)')


# # Across-trial

# ## Parse blocks

# In[ ]:


def parseBlock(df):
    numBlock = 8
    

