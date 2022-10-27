import numpy as np
import pandas as pd
from numpy import var
from numpy import mean
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import glob, re, os
import scipy
import scipy.stats
from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu,ttest_ind
import nltk
from tqdm import tqdm
import pickle as pkl
import spacy
nlp = spacy.load("en_core_web_sm")

def to_int(val):
    if val=='Unknown':
        return float('inf')
    else:
        return int(val)

def latex_float(num, sigfigs=3):
    f = "{0:.%se}" % sigfigs
    float_str = f.format(num)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate Cohen's D
    return (u1 - u2) / s

def format_sig(p1, p2, p, d, title='', sigfigs=3, full_p=False, n1=None, n2=None, scientific=False):
    f = "{:.%sf}" % sigfigs
    prob = f.format(p)
    L = f.format(p1)
    R = f.format(p2)
    D = f.format(d)
    
    if scientific:
        L = latex_float(p1, sigfigs)
        R = latex_float(p2, sigfigs)
    
    stars = ""
    if p<0.05:
        stars += '*'
    if p<0.01:
        stars += '*'
    if p<0.001:
        stars += '*'
        
    if full_p:
        stars += " (p=%s)" % p
        
    sizes = ""
    if n1 and n2:
        sizes = " & %s & %s" % (n1, n2)
    
    if title:
        return "%s %s & %s & %s%s & %s \\\\" % (' '.join(title.split('_')).capitalize(), stars, L, R, sizes, D)
    else:
        return "%s & %s & %s%s & %s \\\\" % (stars, L, R, sizes, D)

def test_binom(d1, d2, alpha=0.05, title="", sigfigs=3, full_p=False, include_n=False):
    n1, n2 = len(d1), len(d2)
    p1, p2 = sum(d1)/n1, sum(d2)/n2
    
    p = (n1*p1 + n2*p2) / (n1 + n2)
    Z = (p1-p2) / np.sqrt(p*(1-p)*((1/n1)+(1/n2)))
    p_value = scipy.stats.norm.sf(abs(Z))*2
    
    D = cohend(d2,d1)
    
    if include_n:
        print(format_sig(p1, p2, p_value, D, title, sigfigs, full_p, n1, n2))
    else:
        print(format_sig(p1, p2, p_value, D, title, sigfigs, full_p))
    
def test(d1, d2, alpha=0.05, title="", sigfigs=3, include_n=False, full_p=False, scientific=False):
    stat, p = mannwhitneyu(d1, d2, alternative='two-sided') #ttest_ind(d1, d2, equal_var=False, nan_policy='omit')
    D = cohend(d2,d1)
    
    if include_n:
        print(format_sig(np.nanmean(d1), np.nanmean(d2), p, D, title, sigfigs, full_p=full_p, n1=len(d1), n2=len(d2), scientific=scientific))
    else:
        print(format_sig(np.nanmean(d1), np.nanmean(d2), p, D, title, sigfigs, full_p=full_p, scientific=scientific))
        
def ks_test(d1, d2, num_bins=None, x_lbl='Feature', y_lbl='CDF', title='CDF Function', 
            plabel='positive', nlabel='negative', alpha=0.05, fn=None, dpi=600, dval=False):
    
    if not num_bins:
        num_bins = min(len(d1), len(d2))
    
    counts_1, bin_edges_1 = np.histogram(d1, bins=num_bins)
    counts_2, bin_edges_2 = np.histogram(d2, bins=num_bins)
    
    cdf_1 = np.cumsum(counts_1)
    cdf_2 = np.cumsum(counts_2)
    
    ks = ks_2samp(d1, d2)
    
    p = ks.pvalue
    D = ks.statistic
    
    if p<alpha:
        plt.rcParams.update({'font.size': 14})
        plt.ylim(-0.1, 1.1)
        plt.gcf().subplots_adjust(bottom=0.20,left=0.15)
        
        plt.plot(bin_edges_1[1:], cdf_1/cdf_1[-1], label=plabel, color='blue')
        plt.plot(bin_edges_2[1:], cdf_2/cdf_2[-1], label=nlabel, color='red')

        plt.xlabel(x_lbl)
        plt.ylabel(y_lbl)
        if dval:
            title = "%s (p=%f, D=%f)" % (title, p, D)
        plt.title(title)
        plt.legend(loc='lower right')
        if fn:
            plt.savefig(fn, dpi=dpi)
        plt.show()
    
    return ks
        
def leaning_label(score):
    if score < -29:
        return 'Extreme Left'
    elif score < -17:
        return 'Left'
    elif score < -5:
        return 'Left Center'
    elif score < 6:
        return 'Least Biased'
    elif score < 18:
        return 'Right Center'
    elif score < 31:
        return 'Right'
    else:
        return 'Extreme Right'
    
def save_regression_plot(x,y,name,x_lbl="MBFC Leaning Score\n(Liberal < 0 < Conservative)", y_lbl="Document Framing\nProportion", sigfigs=3):
    
    font = {'family' : 'normal',
            'size'   : 18}

    plt.rc('font', **font)

    frmt = r'''$r=%(pears)s$
$p=%(proba)s$'''
    
    r,p = stats.pearsonr(x,y)
    f = "{:.%sf}" % sigfigs
    p_str = f.format(p)
    r_str = f.format(r)
    
    plt.scatter(x, y, color='#4c658f')
    ax = sns.regplot(x=x, 
                y=y, 
                line_kws={"color":"#3d3f42","alpha":0.7,"lw":5},
                label= frmt % {'pears': r_str, 'proba': p_str},
                scatter=False
               )
    if x_lbl or y_lbl:
        ax.set(xlabel=x_lbl, ylabel = y_lbl)
    plt.ylim(-0.1, 1.1)
    
    plt.gcf().subplots_adjust(left=0.25, bottom=0.25)
    if not (y_lbl and x_lbl):
        ax.set_yticks([])
        #plt.gcf().subplots_adjust(bottom=0.25)
    
    plt.legend()
    
    if not os.path.exists('img/regression'):
        os.makedirs('img/regression')
    fn ='img/regression/%s_regression.pdf'%name
    print(fn)
    plt.savefig(fn, dpi=600)
    plt.show()