### Masked kalman filter for trajectory data
### Written by S Dey
import xml.etree.ElementTree as ET  
import gpxpy
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime,timedelta
import osmnx as ox
import networkx as nx
import xml.sax
import fiona
import csv
from rtree import index
import math
from math import exp, sqrt
from datetime import datetime
import itertools
from collections import Counter, defaultdict
from pyproj import Proj, transform
from sklearn.neighbors import NearestNeighbors
import scipy
from scipy import spatial
import xml.sax
import statistics
import ast
from os import listdir,path
import time
from numpy import ma
from pykalman import KalmanFilter
#import pywt
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy

plt.rcParams.update({'font.size': 18})

FMT = '%H:%M:%S'
def DTtoFloat(dstring):
    x = datetime.strptime(dstring, FMT)
    return x.hour * 3600 + x.minute * 60 + x.second


def AccPointwise(velsp,t):
    tL = [[]] * len(t) #
    aL = [0] * len(t)
    for i in range(1,len(t)):
        v1 = velsp[i-1]
        v2 = velsp[i]

        dx2 = v2-v1 
        dt2 =  (DTtoFloat(t[i]) - DTtoFloat(t[i-1]))
        try:
            aL[i] = dx2/dt2
        except:
            aL[i] = 0
        
    return np.array(aL)

def VxVy(lon,lat,time): ## return velocities accroding to x,y axis
    t = time
    tL = [[]] * len(t)
    z = [[]] * len(t)#
    #tL[0] = osmTodis(osmE.u[e[0]], osmE.v[e[0]],ogp)[0]
    vLx = [0] * len(t)
    vLy = [0] * len(t)
    for i in range(1,len(t)):
        lat1 = lat[i-1]
        lat2 = lat[i]
        lon1 = lon[i-1]
        lon2 = lon[i]
        dx = (lon2-lon1)
        dy = lat2-lat1
        dt =  (DTtoFloat(t[i]) - DTtoFloat(t[i-1]))
        try:
            vLx[i] = dx/dt
            vLy[i] = dy/dt
        except:
            vLx[i] = 0
            vLy[i] = 0
        z[i] = np.array([lon[i],lat[i],vLx[i],vLy[i]])
    return z


def AnglePointwiseProj(pointsp): # # calculate angle in radians
    #pointsp = list(optdf.point.apply(lambda x: ast.literal_eval(x)))
    lon = []
    lat = []
    for item in pointsp:
        lon.append(item[0])
        lat.append(item[1])
    theta = [0] * len(pointsp)
    for i in range(1,len(pointsp)):
        lat1 = lat[i-1]
        lat2 = lat[i]
        lon1 = lon[i-1]
        lon2 = lon[i]
        dy = lat2-lat1
        dx = lon2-lon1
        the = math.atan2(dy,dx)
        if the < 0:
            the = 2*math.pi + the
            
        theta[i] = the
    angles = np.rad2deg(np.array(theta)) # in degree
    angdif = abs(np.ediff1d(angles))
    angdiff = np.insert(angdif,0,0)
    return angdiff 

def VelocityPointwiseProj(pointsp,t): #t is list of time
    lon = []
    lat = []
    for item in pointsp:
        lon.append(item[0])
        lat.append(item[1])
    
    tL = [[]] * len(t) #
    tL[0] = 0
    #tL[0] = osmTodis(osmE.u[e[0]], osmE.v[e[0]],ogp)[0]
    vL = [0] * len(t)
    for i in range(1,len(t)):
        lat1 = lat[i-1]
        lat2 = lat[i]
        lon1 = lon[i-1]
        lon2 = lon[i]
        dx = sqrt((lat1-lat2)**2 +(lon1-lon2)**2 )
        dt =  (DTtoFloat(t[i]) - DTtoFloat(t[i-1]))
        try:
            vL[i] = dx/dt
        except:
            vL[i] = 0
        tL[i] = dt 
    return np.array(vL),tL #* 3.6


def MaskedKalmanFilter(pointsp,timeList):
    lon = []
    lat = []
    for item in pointsp:
        lon.append(item[0])
        lat.append(item[1])
        
    L = len(timeList)
     # 2-d array 
    x0 = np.array([lon[0],lat[0],0,0]).T #[posx, posy, velx, vely] 
    ## process error in velocity = measurement error in velocity. del_vx = 1/9 m/s del_x = 30m
    P0 = np.array([[50,0,0,0],
                   [0,50,0,0],
                   [0,0,1/81,0],
                   [0,0,0,1/81]]) ##initial process covariance matrix modify ##
##    V = np.array([[100,0,0,0],
##                   [0,100,0,0],
##                   [0,0,1/81,0],
##                   [0,0,0,1/81]]) #Measurement error variance with gaussian noise N(0,R)
    W = np.array([[100,0,0,0],
                   [0,100,0,0],
                   [0,0,1/81,0],
                   [0,0,0,1/81]])    #prediction error variance with gaussian noise N(0,Q)
    x = [[]] * L  #input state
    f_v = [0] * L 

    P = [[]] * L  # process covarinace matrix
    K = [[]] * L    #Kalman gain
    x[0] = x0
    P[0] = P0
    zk = VxVy(lon,lat,timeList) #calculate velocity along x axis and y axis

    t_m = [[]] * (L-1)

    for k in range(1,L):
        ti = timeList[k]
        ti1 = timeList[k-1]
        dt = (DTtoFloat(ti) - DTtoFloat(ti1))
        A = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]) #
        t_m[k-1] = A 


    measurements = np.asarray(pointsp)#np.asarray([(399,293),(403,299),(409,308),(416,315),(418,318),(420,323),(429,326),(423,328),(429,334),(431,337),(433,342),(434,352),(434,349),(433,350),(431,350),(430,349),(428,347),(427,345),(425,341),(429,338),(431,328),(410,313),(406,306),(402,299),(397,291),(391,294),(376,270),(372,272),(351,248),(336,244),(327,236),(307,220)])
    initial_state_mean = [lon[0],
                      lat[0],
                      0,
                      0]

    transition_matrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]) #np.array([[1,5,0,0],[0,1,0,0],[0,0,1,5],[0,0,0,1]])

    observation_matrix = [[1, 0, 0, 0],
                      [0, 1, 0, 0]] ## H 

    observation_covariance = [[30,0], ## V
                          [0,30]]

    kf2 =  KalmanFilter(transition_matrices = transition_matrix,
                  observation_matrices = observation_matrix, ## H
                  observation_covariance = observation_covariance, #V
                  initial_state_mean = initial_state_mean, transition_covariance = W, # 
                  initial_state_covariance = P0 )
    delTime = VelocityPointwiseProj(pointsp,timeList)[1]
    cumTime = np.array(delTime).cumsum()
    
    if np.any(cumTime < 0): ## if there are negative values in cumulative time, there is an error in the data       
        print("--Dataset has wrongly ordered time stamps. Please rectify.--")
        return []
    else:
        TotalTime = sum(delTime)
        all_measure = [(0,0)] * (TotalTime+1)
        Modi_Measure = np.array(all_measure)
        Modi_Measure[cumTime] = pointsp
        To_mask = np.delete(np.arange(0,TotalTime+1), cumTime) ### Indexes where missing values of measurements are,
        ## to be masked on the missing values of measurements
        
        missing_measure = ma.asarray(Modi_Measure)

        missing_measure[To_mask] = ma.masked

        kf2 = kf2.em(missing_measure,n_iter=5) 
        (state_means, state_covari) = kf2.smooth(missing_measure)
        vfx_all = state_means[:, 2]
        vfy_all = state_means[:, 3]
        vf_all = np.sqrt(np.square(vfx_all) + np.square(vfy_all))
        vf = []
        xf = [state_means[0][0]]
        yf = [state_means[0][1]]
        vf.append(vf_all[0])
        for idx in range(1,len(cumTime)):
            prevTime = cumTime[idx-1]
            nextTime = cumTime[idx]
            time_gap = delTime[idx]
            if len(vf_all[prevTime:nextTime]) == time_gap:
                #print("yes")
                sum_of_sampled_velocity = sum(vf_all[prevTime:nextTime])
                #average_velocity = sum_of_sampled_velocity/time_gap
                point_velocity = vf_all[nextTime]
                
                vf.append(point_velocity)
                xf.append(state_means[nextTime][0])
                yf.append(state_means[nextTime][1])
            else:
                print('no')
        
        af = AccPointwise(vf,timeList)
    ####    vf is velocity filtered, af is acceleration filtered, angf angles filtered
##        anglesFiltered = np.rad2deg(AnglePointwiseProj(list(zip(xf,yf))))
##        anglediffFiltered = np.ediff1d(anglesFiltered).tolist()
##        anglediffFiltered.insert(0,0)
        angf = AnglePointwiseProj(list(zip(xf,yf))) # angles in radians,change to degree by np.rad2deg(np.array(theta))

   

    return xf,yf,vf,af,angf

##1360 to 1365 <numTr value>
