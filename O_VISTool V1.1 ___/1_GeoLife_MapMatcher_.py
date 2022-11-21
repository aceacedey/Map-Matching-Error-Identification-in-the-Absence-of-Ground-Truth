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
from shapely.geometry import shape, Point, mapping
from shapely.ops import linemerge
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
import xml.etree.ElementTree as ET
from scipy import spatial
import xml.sax
import statistics


import os
from os import listdir
from os.path import isfile, join
import time
import mmHmmSDEY as mm
import utm
import MaskedKalmanFilter as mkf

FMT = '%H:%M:%S'
FMTD = '%D/%M/:Y'
def path_cost(G, path):
    return sum([G[path[i]][path[i+1]][0]['length'] for i in range(len(path)-1)])

def osmTodis(o1,o2,og):#,omsN,osmE): # takes two osmids, return two lists, distance time velocity etc.
    #tl = [t1,t2]
    #tdelta = datetime.strptime(t2, FMT) - datetime.strptime(t1, FMT)
    try:
        path = nx.shortest_path(og,o1,o2,weight='length')
    except:
        path = []
    if path:dist = sum([og[path[i]][path[i+1]][0]['length'] for i in range(len(path)-1)])
    else: dist = 1000000 ## almost infinite for no path
    #vel = []
    return path,dist

    
def removeListfromListNUMPY(b): ## remove lists from a list
    #print(b)
    a=[]
    for i,item in enumerate(b):
        if type(item) == list: # or type(item) == list:
            a.append(i)
    return np.delete(b,a)


def OSMmidPoint(u,v):
    return ( (osmN.x[u] + osmN.x[v])/2 , (osmN.y[u] + osmN.y[v])/2 )

def DistPoint(p1,p2):
    return sqrt((p1[0]-p2[0])**2 +(p1[1]-p2[1])**2 )

def AreaTriangle(oe,p):
    #print(oe)
    x1 = osmN.x[osmE.u[oe]]
    y1 = osmN.y[osmE.u[oe]]
    x2 = osmN.x[osmE.v[oe]]
    y2 = osmN.y[osmE.v[oe]]
    a = np.array([[x1,y1,1],[x2,y2,1],[p[0],p[1],1]])
    #print(a)
    return abs(np.linalg.det(a))

def OrthoDist(oe,p):
    osmEt = osmE.set_index('osmid')
    
    x1 = osmN.x[osmEt.u[oe]] # osmN[osmN.osmid == oe].x.values[0] ##
    y1 = osmN.y[osmE.u[oe]]
    x2 = osmN.x[osmE.v[oe]]
    y2 = osmN.y[osmE.v[oe]]
    A = np.array([x1,y1])
    B = np.array([x2,y2])
    C = np.array(p)
    
    d = np.linalg.norm(np.cross(B - A, C - A))/np.linalg.norm(B - A)
    return d,[x1,y1],[x2,y2]

def DTtoFloat(dstring):
    x = datetime.strptime(dstring[0], FMT)
    return x.hour * 3600 + x.minute * 60 + x.second


def Velocity(optdf):
    e = optdf.edge.values
    t = optdf.time.values
    tL = [[]] * len(e) #
    #tL[0] = osmTodis(osmE.u[e[0]], osmE.v[e[0]],ogp)[0]
    vL = [0] * len(e)
    dx = 0
    dt = 0
             
    for i in range(1,len(e)):
        if e[i] == e[i-1]:
            dx = osmE.length[e[i]]
            dt = dt + (DTtoFloat(t[i]) - DTtoFloat(t[i-1]))
            #vL[i] = vL[i-1]
        elif e[i] != e[i-1] and i != 1:
            try:vL[i-1] = dx/dt
            except: vL[i-1] = 0.0
            tL[i-1] = osmTodis(osmE.u[e[i-1]], osmE.v[e[i-1]],ogp)[0]
            o1 = osmE.v[e[i-1]]
            o2 = osmE.v[e[i]]
            op = osmTodis(o1,o2,ogp)
            dx = op[1]### find the shortest path between e[i] and e[i-1]
            dt = (DTtoFloat(t[i]) - DTtoFloat(t[i-1]))
           
        else:
            o1 = osmE.v[e[i-1]]
            o2 = osmE.v[e[i]]
            op = osmTodis(o1,o2,ogp)
            dx = op[1]### find the shortest path between e[i] and e[i-1]
            dt = (DTtoFloat(t[i]) - DTtoFloat(t[i-1]))
            vL[i] = dx/dt
            tL[i] = op[0]
                
    try:vL[i] = dx/dt
    except: vL[i] = 0.0
    tL[i] = osmTodis(osmE.u[e[-1]], osmE.v[e[-1]],ogp)[0]
    
    return np.array(vL) * 3.6,tL


startTime = datetime.now()
start = time.process_time()
path = "Geolife_Raw_CWTrajectories"
allFiles =  listdir(path)

time_per_df = pd.DataFrame(columns=['dataset','MM_','MM_v','MM_e','MM_ae'])

if True:
## To save some time, we ignored the Geolife CW data those can not be map-matched at all by the algorithm
    EligibleFiles = ['DataSet_15.csv', 'DataSet_16.csv',
                     'Dataset_2.csv', 'DataSet_20.csv', 'DataSet_26.csv', 'DataSet_27.csv', 
                     'DataSet_34.csv', 'DataSet_35.csv','DataSet_37.csv', 'DataSet_38.csv','DataSet_39.csv',
                     'DataSet_4.csv','DataSet_41.csv', 'DataSet_43.csv', 'DataSet_45.csv', 'DataSet_46.csv', 'DataSet_47.csv','DataSet_49.csv',
                    'DataSet_50.csv','DataSet_51.csv', 'DataSet_53.csv','DataSet_57.csv',
                     'DataSet_60.csv', 'DataSet_63.csv', 'DataSet_66.csv', 'DataSet_67.csv', 'DataSet_68.csv',
                 'Dataset_7.csv', 'DataSet_70.csv', 'DataSet_71.csv','DataSet_72.csv', 'DataSet_76.csv', 'DataSet_77.csv']#[ffile for ffile in filenames]
##
##    NoMatchedFiles = set(allFiles).intersection(set(EligibleFiles))
##    for del_file in list(NoMatchedFiles):
##        del_file_path = str(path) + "/" + str(del_file)
##        #os.remove(del_file_path)
##        print(del_file_path)
    #filenames = EligibleFiles 

print('Starting')
filenames = EligibleFiles #listdir(path)
print(len(filenames))

for i_row,file in enumerate(filenames):#eL:#EligibleFiles:
    file_start = time.process_time()
    fPath = str(path) + "/" + str(file)
    print(fPath)
    time_per_df.loc[i_row,'dataset'] = file ## insert file name
    
    rawdf = pd.read_csv(fPath)

    pointsp = rawdf.apply(lambda x: utm.from_latlon(x.lat,x.lon)[0:2],axis=1).to_list() # rawdf['projectedLonLat'] ## utm package take lat,lon and convert to lon, lat
    rawdf['pointsp'] = pointsp
    lon, lat = list(rawdf.lon.values), list(rawdf.lat.values)
    points = list(zip(lon,lat))
    bbox = [max(lat),min(lat), max(lon), min(lon)] ##[maxLat,minLat,maxLon,minLon]
    og = ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3],truncate_by_edge=True,retain_all=False,simplify=True)
    ogp = ox.projection.project_graph(og)
    ##gb = ox.add_edge_bearings(ogp)
    #ogp = ogp.to_undirected() ## fo undirected graph
    osmNg = ox.graph_to_gdfs(ogp,edges=False)
    osmEg = ox.graph_to_gdfs(ogp,nodes=False)
    osmN = osmNg.reset_index()
    #osmN.set_index('osmid',inplace=True,keep=True)
    osmE = osmEg.reset_index()
    #osmE.set_index('osmid',inplace=True)
    #osmE['midpoint'] = osmE.apply(lambda x: OSMmidPoint(x.u,x.v),axis=1)
    
    # Endpoint coordinates of segment and its length
    seg_info = mm.getSegmentInfo(osmN,osmE)
    # Geometry of street segments ## is a dictionary with edge id and segment geom
    seg_geom = mm.getSegmentShapes(osmE)
    # Transform street network into a primal graph
    G = ogp#mm.getNetworkGraph(og, seg_info[1])
    # Coordinates of endpoints of each street segment
    endpoints = seg_info[0]
    # Length of each street segment
    length = seg_info[1]
    # Create RTREE index for the street network
    idx = mm.buildRTree(osmE)
    ##print(datetime.now() - startTime)
    trajdf = pd.DataFrame(columns=['date','d_id','s_id','start','end','osmidS','osmidE','velocity','distance','speedlimit','trajectory','points'])
    Output = mm.mapMatch(pointsp,list(rawdf.time), seg_info, G, idx, seg_geom,osmN,osmE, 100, 100, 50)
    if len(Output) > 10:
        ##print('Got Output')
        optdf = Output.reset_index(drop=True)
        #optdf = odf# pd.DataFrame(odf.values,columns=['time'],index=odf.index.values)
        optdf['u'] = osmE.u[optdf.edge.values].values
        optdf['v'] = osmE.v[optdf.edge.values].values
        optdf['distance'] = osmE.length[optdf.edge.values].values
        optdf['ListGeom'] = optdf.edgeGeom.apply(lambda x:list(zip(x.xy[0].tolist(),(x.xy[1].tolist()))))
        optdf['MMfrom'] = optdf.ListGeom.apply(lambda x: list(x[0]))
        optdf['MMto'] = optdf.ListGeom.apply(lambda x: list(x[-1]))
        optdf['trajectory'] = Velocity(optdf)[1]
        l = [0] * (len(optdf) - 4)
        Newbbox = bbox + l
        optdf['Bbox'] = Newbbox ## saving bouing box        
        curName = "MapMatcched Trajectories Geolife/" + 'MM_geolife_' + str(file)
        optdf.to_csv(curName)
    #print(curName)
    file_end = time.process_time()
    process_time = file_start - file_end

    time_per_df.loc[i_row,'MM_'] = process_time ## insert file name
    print('-------')
    #break
        
# your code here
End_time = time.process_time()
print(End_time - start)
time_per_df.to_csv('time_per_df_MM_only.csv')
