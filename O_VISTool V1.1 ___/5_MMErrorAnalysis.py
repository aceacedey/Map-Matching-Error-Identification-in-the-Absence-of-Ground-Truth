#Frechet distance calculation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
import osmnx as ox
import shapely
from shapely.geometry import Point, LineString
from shapely import geometry, ops
import math
import utm
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import Counter
from pylab import rcParams
from os import listdir
from os.path import isfile, join
import time
from datetime import datetime,timedelta
from numpy import ma
from sklearn.metrics import confusion_matrix
from sklearn import svm, metrics
import pywt
from scipy.stats import entropy


plt.rcParams.update({'font.size': 20})
plt.rcParams['pdf.fonttype'] = 42

def CallC(m, i, j, p, q):

   if m[i, j] > -1:
       return m[i, j]
   elif i == 0 and j == 0:
       m[i, j] = np.linalg.norm(p[i]-q[j])
   elif i > 0 and j == 0:
       m[i, j] = max(CallC(m, i-1, 0, p, q), np.linalg.norm(p[i]-q[j]))
   elif i == 0 and j > 0:
       m[i, j] = max(CallC(m, 0, j-1, p, q), np.linalg.norm(p[i]-q[j]))
   elif i > 0 and j > 0:
       m[i, j] = max(
           min(              
               CallC(m, i-1, j, p, q),
               CallC(m, i-1, j-1, p, q),
               CallC(m, i, j-1, p, q)
           ),
           np.linalg.norm(p[i]-q[j])
           )
   else:
       m[i, j] = float('inf')

   return m[i, j]


def frdist(p, q): #q = [[*row] for row in list(x.edgeGeom.coords)]
   p = np.array(p, np.float64)
   q = np.array(q, np.float64)
   len_p = len(p)
   len_q = len(q)
   if len_p == 0 or len_q == 0:
       raise ValueError('Input curves are empty.')

   #if len_p != len_q or len(p[0]) != len(q[0]):
    #   raise ValueError('Input curves do not have the same dimensions.')

   m = (np.ones((len_p, len_q), dtype=np.float64) * -1)
   dist = CallC(m, len_p-1, len_q-1, p, q)
   return dist


def PlotPointsAfterClustering(df1,ogp,errorL,geomL):
#if 1 == 1: 
   #plt.rcParams.update({'font.size': 20})
   fig, ax = ox.plot_graph(ogp, show=False, close=False)
   
   lon = []
   lat = []
   clusterColour = ['g','r','b','k'] ## for 3 clusters
   modeColour = {'car':'m','walk':'c'}
   #### uncomment for colouring based on clusters
   unClus = list(df1.cluster.unique())
  
   for item in unClus:
      tdf = df1[df1.cluster == item]
      lon = tdf.gps_x.to_list()
      lat = tdf.gps_y.to_list()
      #ax.plot(lon,lat,c=clusterColour[item],marker='*', linestyle=' ')
      ax.plot(lon,lat,c='b',marker='o', linestyle=' ',markersize=8)
   osmN,osmE = ox.graph_to_gdfs(ogp,edges=True,nodes=True)
   #df1['Geom'] = df1.edge.apply(lambda x: osmE.loc[x]['geometry'])
   mmdf = df1[df1.trajectory != '[]']
   #mmdf.apply(lambda x: ax.plot(x.edgeGeom.xy[0].tolist(),x.edgeGeom.xy[1].tolist(),c=clusterColour[x.cluster], linestyle='--'),axis=1)

   for idx,item in enumerate(geomL):
      col =  clusterColour[errorL[idx]]
      #col = 'g' ## for no error graph generation
      xL = item.xy[0].tolist()
      yL = item.xy[1].tolist()
      ax.plot(xL,yL,c=col,marker=' ', linestyle='--',linewidth=3) ## cluster colour according to cluster value
      #for an in list(item.coords):
      mid = ((xL[1] + xL[0])/2,(yL[1] + yL[0])/2)
      ax.text(mid[0], mid[1], str(idx),color='black', fontsize=30)
      #ax.annotate(idx,mid)
   plt.show()

def fre(x,y,x1,y1,x2,y2):
    p=[(x,y)]
    q=[[x1,y1],[x2,y2]]
    return frdist(p,q)

def GetClosestPointx(Ax,Ay,Bx,By,Px,Py):
  a_to_p = [Px - Ax, Py - Ay]     # Storing vector A->P
  a_to_b = [Bx - Ax, By - Ay]     # Storing vector A->B

  atb2 = a_to_b[0]**2 + a_to_b[1]**2  # **2 means "squared"
                                      #   Basically finding the squared magnitude
                                      #   of a_to_b

  atp_dot_atb = a_to_p[0]*a_to_b[0] + a_to_p[1]*a_to_b[1]
                                      # The dot product of a_to_p and a_to_b

  t = atp_dot_atb / atb2              # The normalized "distance" from a to
                                      #   your closest point
    
  return Ax + a_to_b[0]*t
                                      # Add the distance to A, moving
                                      #   towards B
def GetClosestPointy(Ax,Ay,Bx,By,Px,Py):
  a_to_p = [Px - Ax, Py - Ay]     # Storing vector A->P
  a_to_b = [Bx - Ax, By - Ay]     # Storing vector A->B

  atb2 = a_to_b[0]**2 + a_to_b[1]**2  # **2 means "squared"
                                      #   Basically finding the squared magnitude
                                      #   of a_to_b

  atp_dot_atb = a_to_p[0]*a_to_b[0] + a_to_p[1]*a_to_b[1]
                                      # The dot product of a_to_p and a_to_b

  t = atp_dot_atb / atb2              # The normalized "distance" from a to
                                      #   your closest point
    
  return Ay + a_to_b[1]*t

#Calculation of orthogonal distance

def orth(x,y,x1,y1,x2,y2):
    p1=np.array([x1,y1])
    p2=np.array([x2,y2])
    p3=np.array([x,y])
    d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
    return abs(d)

def leng(x1,y1,x2,y2):
    return math.sqrt(((x2-x1)**2) + ((y2-y1)**2))

def div(x,y):
    return x/y

def AreaPloy(aList):
   #aList = [(2, 2), (1, 5), (2, 6)]
   aPoly = shapely.geometry.Polygon(aList)
   Area = aPoly.area
   return Area

def Odist(P,Multi):
   line = LineString(Multi)
   point = Point(P)
   #line.within(point)  # False
   dis = line.distance(point)  # 7.765244949417793e-11
   #line.distance(point) < 1e-8 

   return dis

def DTtoFloat(dstring):
   FMT = '%H:%M:%S'
   x = datetime.strptime(dstring, FMT)
   return x.hour * 3600 + x.minute * 60 + x.second
    
def TimeDiff(t):
    #.apply(lambda x: ast.literal_eval(x)[0]).values.tolist() 
    tL = [[]] * len(t) #
    tL[0] = 0
    for i in range(1,len(t)):
        dt =  (DTtoFloat(t[i]) - DTtoFloat(t[i-1]))
        tL[i] = dt 
    return tL #* 3.6


def PlotSubset(x,y,ylabel):
    idx = np.argsort(y)
    sx = np.array(x)[idx]
    sy = np.array(y)[idx]
    
    poix = list(range(1,len(sx)+1))
    fig, ax = plt.subplots()
    ax.plot(poix,sx, c='red',marker='$L$', ms=10, linestyle='--',label=lg1)
    #for an in x1: 
        #ax.annotate(an,poi[an])
    ax.plot(poix,sy,c='blue',marker='$S$', ms=10, linestyle='--',label=lg2)
    
    plt.xlabel(xlabel)
    #plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(1, len(x)+1))
    ax.legend(loc='lower right', frameon=False)
    plt.xticks(poix, np.array(xticks)[idx], rotation=90)
    plt.grid(True)
    #plt.ylim(0,1)
    plt.show()


    
#startTime = datetime.now()
#path = "MapMatcched Trajectories Geolife 1"
path = "to split"
filenames = listdir(path)

EligibleFiles = [ffile for ffile in filenames]
#drive_filter = ox.core.get_osm_filter('drive')
print('now')
acGBdwt = []

acKM = []
acKMdwt = []
acGMM = []
acGMMdwt = []
acHD = []
acHDdwt = []

fn = []

trainMean = []
trainStd = []
dci = []
DS = []
te = []
qMeanStdTest = []
Entropy = []

CMKM = np.zeros((2,2))
CMGMM = np.zeros((2,2))
CMHD = np.zeros((2,2))

def show_metrics(tp,fp,tn,fn):
    # True positive rate (sensitivity or recall)
    tpr = tp / (tp + fn)
    # False positive rate (fall-out)
    fpr = fp / (fp + tn)
    # Precision
    precision = tp / (tp + fp)
    # FOR false omission rate
    FOR = fn / (fn+tn) 
    # True negatvie tate (specificity)
    tnr = 1 - fpr
    # F1 score
    f1 = 2*tp / (2*tp + fp + fn)
    # ROC-AUC for binary classification
    auc = (tpr+tnr) / 2
    # MCC
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    print("True positive: ", tp)
    print("False positive: ", fp)
    print("True negative: ", tn)
    print("False negative: ", fn)

    print("True positive rate (recall): ", tpr)
    print("False positive rate: ", fpr)
    print("Precision: ", precision)
    print("false omission rate: ", FOR)
    print("True negative rate: ", tnr)
    print("F1: ", f1)
    print("ROC-AUC: ", auc)
    print("MCC: ", mcc)

#stI = EligibleFiles.index("20080428134352.csv")
#print(stI)
import MaskedKalmanFilter as mkf
from scipy.spatial.distance import cdist
size = 600
if 1 == 1:
   #mdf = pd.read_csv('man.csv',index_col=0)
   mdf = pd.read_csv('valdf1.csv',index_col=0)
   #mdf['True negative'] =
   #mdf['TP'] = mdf.totalSegment - mdf[['ObservedError','errorEstimatedCorrectly']].max(axis=1)
   mdf['TP'] = mdf.totalSegment - mdf.EstimatedError - mdf.errorNotestimated# '#]].max(axis=1)
   tp = mdf.TP.sum()
   fp = mdf.errorEstimatedIncorrectly.sum()
   fn = mdf.errorNotestimated.sum()
   tn = mdf.errorEstimatedCorrectly.sum()
   show_metrics(tp,fp,tn,fn)
   mdf['MM'] = (mdf.ObservedError)/mdf.totalSegment * 100
   mdf['PM'] = (mdf.ObservedError - mdf.errorEstimatedCorrectly )/mdf.ObservedError * 100
   #mdf['PM'] = (mdf.ObservedError - mdf.errorEstimatedCorrectly )/mdf.totalSegment  * 100
   mdf = mdf.sort_values(['traj'])
   mdf.reset_index(inplace=True)
   mdf.index = np.arange(1, len(mdf)+1)
   #print(mdf.round(2).drop('EstimatedError',axis=1).to_latex())
   print(mdf.drop(columns=['traj']).round(2).to_latex())
##   plt.scatter(mdf.MM.to_list(),mdf.PM.to_list(),c='red',marker='*')
##   plt.xlabel('Percentage of error in map-matching')
##   plt.ylabel('Percentage of error in estimation')
##   #plt.xticks(np.arange(1, len(x)+1))
##   #ax.legend(loc='lower right', frameon=False)
##   #plt.xticks(poix, np.array(xticks)[idx], rotation=90)
##   plt.grid(True)
##   #plt.ylim(0,1)
##   plt.show()

##if 1 == 1:
##   plt.rcParams.update({'font.size': 22})
##   fig, ax1 = plt.subplots()
##   import numpy.polynomial.polynomial as poly
##
##   #y1 = dci
##   x = mdf.MM.to_list()
##   y2 = mdf.PM.to_list()
##   #y2 = Entropy
##   idxs = np.argsort(x)
##   y1 = np.sort(x)
##   y2 = [y2[item] for item in idxs]
##
##   ax1.plot(y1,y2,'ro', ms=8)
##
##   coefs = poly.polyfit(y1, y2, 2)
##   xs = np.linspace(0,max(y1), 2 * len(x))
##   ffit = poly.polyval(xs, coefs)
##   ax1.plot(xs, ffit,'b', ls='--',lw=3,label='Second order polynomial fit')
##   ##     coefs = poly.polyfit(y1[0:-1], y2[0:-1], 3)
##   ##    xs = np.linspace(0,max(y1), 2 * len(dci))
##   ##    fit = poly.polyval(xs, coefs)
##   ##    ax1.plot(xs, ffit,'b', lw=3,label='3rd order polynomial fit')
##   ## 
##   ax1.set_xlabel(r'Percentage of error in map-matching ($\epsilon_{MM}$)')
##   ax1.set_ylabel(r'Percentage of error in model estimation ($\epsilon_{PM}$)')
##   ax1.legend(loc='upper right', frameon=False)
##   plt.grid(True)
##   plt.show()

if 1 == 1: ##linear reg fit
   from sklearn.linear_model import LinearRegression 
   from sklearn.metrics import mean_squared_error, r2_score
   
   plt.rcParams.update({'font.size': 22})
   fig, ax1 = plt.subplots()

   #y1 = dci
   x = mdf.MM.to_list()
   y2 = mdf.PM.to_list()
   #y2 = Entropy
   idxs = np.argsort(x)
   y1 = np.sort(x)
   y2 = [y2[item] for item in idxs]

   ax1.plot(y1,y2,'ro', ms=8)
   y1 = np.reshape(y1,(1,-1)).T
   y2 = np.reshape(y2,(1,-1)).T
   reg = LinearRegression()
   reg.fit(y1,y2)
   reg.score(y1, y2)
   slope = reg.coef_[0][0]
   intercept = reg.intercept_[0]
   x_vals = np.array(ax1.get_xlim())
   y_vals = intercept + slope * x_vals
   ax1.plot(x_vals, y_vals, 'b', ls='--',lw=3,label='Linear regression line')
   #ax1.plot(xs, ffit,'b', ls='--',lw=3,label='Second order polynomial fit')
   ax1.set_xlabel(r'Percentage of error in map-matching ($\epsilon_{MM}$)')
   ax1.set_ylabel(r'Percentage of error in model outcome($\epsilon_{PM}$)')
   ax1.legend(loc='upper right', frameon=False)
   plt.grid(True)
   plt.show()
   y_pred = reg.predict(y1)
   print('Mean squared error: %.2f' %mean_squared_error(y2, y_pred))
# The coefficient of determination: 1 is perfect prediction
   print('Coefficient of determination: %.2f'%r2_score(y2, y_pred))
   


mdfT =(mdf.ObservedError.sum() - mdf.errorEstimatedCorrectly.sum() )/mdf.ObservedError.sum() * 100
##def show_metrics(y_true, y_score):
##    # True positive
##    tp = np.sum(y_true * y_score)
##    # False positive
##    fp = np.sum((y_true == 0) * y_score)
##    # True negative
##    tn = np.sum((y_true==0) * (y_score==0))
##    # False negative
##    fn = np.sum(y_true * (y_score==0))
