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
from pyproj import Transformer
import networkx as nx
import itertools
from networkx.algorithms import bipartite

plt.rcParams.update({'font.size': 20})

def c(m, i, j, p, q):

   if m[i, j] > -1:
       return m[i, j]
   elif i == 0 and j == 0:
       m[i, j] = np.linalg.norm(p[i]-q[j])
   elif i > 0 and j == 0:
       m[i, j] = max(c(m, i-1, 0, p, q), np.linalg.norm(p[i]-q[j]))
   elif i == 0 and j > 0:
       m[i, j] = max(c(m, 0, j-1, p, q), np.linalg.norm(p[i]-q[j]))
   elif i > 0 and j > 0:
       m[i, j] = max(
           min(              
               c(m, i-1, j, p, q),
               c(m, i-1, j-1, p, q),
               c(m, i, j-1, p, q)
           ),
           np.linalg.norm(p[i]-q[j])
           )
   else:
       m[i, j] = float('inf')

   return m[i, j]

def frdist(p, q):
   p = np.array(p, np.float64)
   q = np.array(q, np.float64)
   len_p = len(p)
   len_q = len(q)
   if len_p == 0 or len_q == 0:
       raise ValueError('Input curves are empty.')

   #if len_p != len_q or len(p[0]) != len(q[0]):
    #   raise ValueError('Input curves do not have the same dimensions.')

   m = (np.ones((len_p, len_q), dtype=np.float64) * -1)
   dist = c(m, len_p-1, len_q-1, p, q)
   return dist


def PlotPointsAfterClustering(df1):
#if 1 == 1: 
   bbox = df1.Bbox.to_list()[0:4]
   pointsp = df1.point.apply(lambda x: ast.literal_eval(x)).to_list()
   og = ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3],truncate_by_edge=True,network_type="drive")
   #transformer = Transformer.from_crs('epsg:4326', 'epsg:28355')
   ogp = ox.project_graph(og) ##
   #osmN = ox.graph_to_gdfs(ogp,edges=False)
   #osmE = ox.graph_to_gdfs(ogp,nodes=False)
   fig, ax = ox.plot_graph(ogp, show=False, close=False,node_alpha=.2,edge_alpha=.3, bgcolor='w',node_color='b', node_size=1)
   
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
      #ax.plot(lon,lat,c=clusterColour[item],marker='o', linestyle=' ',markersize=5,alpha=0.1)
      ax.plot(lon,lat,c='b',marker='o', linestyle=' ',markersize=5,alpha=0.5)
   #### uncomment for colouring based on Travel modes
##   unTr = list(df1.TrMode.unique())
##   for item in unTr:
##      tdf = df1[df1.TrMode == item]
##      lon = tdf.gps_x.to_list()
##      lat = tdf.gps_y.to_list()
##      clr = modeColour[item]
##      ax.plot(lon,lat,c=clr,marker='*', linestyle=' ')
      
   osmN,osmE = ox.graph_to_gdfs(ogp,edges=True,nodes=True)
   #df1['Geom'] = df1.edge.apply(lambda x: osmE.loc[x]['geometry'])
   mmdf = df1[df1.trajectory != '[]']
   mmdf.apply(lambda x: ax.plot(x.edgeGeom.xy[0].tolist(),x.edgeGeom.xy[1].tolist(),c=clusterColour[x.edgeColour], linestyle='-',marker='o',linewidth=3),axis=1)

   #mmdf.apply(lambda x: ax.plot(x.edgeGeom.xy[0].tolist(),x.edgeGeom.xy[1].tolist(),c='magenta', linestyle='--',marker=' ',linewidth=3),axis=1)

   geomL = mmdf.edgeGeom.to_list()
   errorL = df1.cluster.to_list()
   for idx,item in enumerate(geomL):
      #col =  clusterColour[errorL[idx]]
      #col = 'g' ## for no error graph generation
      xL = item.xy[0].tolist()
      yL = item.xy[1].tolist()
      #ax.plot(xL,yL,c=col,marker=' ', linestyle='--',linewidth=3) ## cluster colour according to cluster value
      #for an in list(item.coords):
      mid = ((xL[1] + xL[0])/2,(yL[1] + yL[0])/2)
      ax.text(mid[0], mid[1], str(idx),color='black', fontsize=15)
      #ax.annotate(idx,mid)
   plt.show()
   

def fre(x,y,x1,y1,x2,y2):
    p=[[x,y]]
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

def FindDisconnectedSegments(mg): ## mg is the mapmatched graph
#if 1 == 1:
   wcg = list(nx.connected_components(mg))
   disconNodes = [list(c) if len(c) <=3 else [] for c in sorted(wcg, key=len, reverse=True)]
   fromToNodes = list(filter(None,disconNodes)) ### give a flat list from the sublists)
   disconEdges = []
   for item in fromToNodes:
      if len(item) > 1:
         possDisconEdges = list(itertools.combinations(item,2))
         for i in possDisconEdges:
            if mg.get_edge_data(i[0],i[1]): #[0]['edge']:
               disconEdges.append(mg.get_edge_data(i[0],i[1])[0]['edge']) 
      else:
         curEdge = mg.get_edge_data(item[0],item[0])[0]['edge']
         disconEdges.append(curEdge)

   return disconEdges
   #mg = nx.from_pandas_edgelist(d9, source='u', target='v', edge_attr='edgeGeom', create_using=nx.MultiDiGraph())
   #d = nx.weakly_connected_components(mg) list(nx.connected_components(mg))
   #[c if len(c) <=3 else {} for c in sorted(nx.connected_components(mg), key=len, reverse=True)]

def FindClaws(mg): ## this function flind hanging segments of claws: 
#if 1 == 1:    
    d = mg.degree()
    listWith3DegNodes = [item[0] for item in d  if item[1] > 2]
    clawEdges = []
    for node in listWith3DegNodes: # neighbourNodes are the neighbour of node
        neighbourNodes = [n for n in mg.neighbors(node) if mg.degree(n) == 1]  ## find all the neighbour nodes of a node of degree three 
             ## find which of the neighbour has degree less than 2 i.e. terminates i.e 
        if len(neighbourNodes) > 0:#== 1:
            for nextNode in neighbourNodes:
                clawEdges.append(mg.get_edge_data(node,nextNode)[0]['edge'])
    return clawEdges
   
def CheckDegrees(u,v,mgc):
   if mgc.degree(u) == 2 and mgc.degree(v) == 2:
      return 0
   else:
      return 1
def CalculateSegmentSpeed(d9):
#if 1 == 1:
   speedseg = []
   dfList = [x for _, x in d9.groupby('Length_of_road_segment')]
   for df in dfList:
      length = df['Length_of_road_segment'].values[0]
      sp = len(df) * [length/df.deltime.sum()]
      speedseg = speedseg + sp
   return speedseg

#startTime = datetime.now()
path = "MapMatcched Trajectories Geolife 1"
#path = "MapMatcched Trajectories Geolife"
#path = "to split"
#path = "Trajectories used for error identification"

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
#stI = EligibleFiles.index("20080428134352.csv")
#print(stI)
import MaskedKalmanFilter as mkf
from scipy.spatial.distance import cdist


#### create a datafram with dataset number, total segment, and number of estimated red segments ::
### for that create a dictionary using d9 , doing it here
valdf = pd.DataFrame()


totalGNSSPoints = 0
for file in EligibleFiles:
   fPath = str(path) + "/" + str(file)

   file_start = time.process_time()
   df1 = pd.read_csv(fPath)
   ###########
   rawdf = df1.reset_index(drop=True)
   
   pointsp = list(rawdf.point.apply(lambda x: ast.literal_eval(x)))
   time = rawdf.time.apply(lambda x: ast.literal_eval(x)).sum()
   #delete points those have impossible velocities before applying KF

   #rawdf['ModeChanges'] = (rawdf.observed.diff(1)!= 0).astype('int').cumsum()
   delTs = TimeDiff(time)
   rawdf["deltime"] = delTs
   kfout = mkf.MaskedKalmanFilter(pointsp,time)
   if kfout:
      fpoints = list(zip(kfout[0],kfout[1]))
   rawdf['pointsf'] = fpoints
   vf = kfout[2]
   accelP = kfout[3]
   rawdf['velocityFiltered'] = vf
   rawdf['accelFiltered'] = accelP
   rawdf['AngleFiltered'] = kfout[4]
   dist_Mat = cdist(np.array(pointsp),np.array(fpoints))
   rawdf['DistDiff']  = dist_Mat.diagonal()
##   if 'walk' in set(df1.TrMode.to_list()):
##      print(file)
###if 1 == 1:
   d2=rawdf.copy()
   #d2.head()
   d2['point'] = d2['point'].str.strip('()')
   d2[['gps_x','gps_y']] = d2.point.str.split(expand=True)
   d2['gps_x'] = d2['gps_x'].str.strip(',')
   d2['gps_x']=d2['gps_x'].astype(float)
   d2['gps_y']=d2['gps_y'].astype(float)
##   d2['MMfrom'] = d2['MMfrom'].str.strip('[]')
##   d2['MMto'] = d2['MMto'].str.strip('[]')
##   d2[['MMfrom_x','MMfrom_y']] = d2.MMfrom.str.split(expand=True)
##   d2['MMfrom_x'] = d2['MMfrom_x'].str.strip(',')
##   d2['MMfrom_x']=d2['MMfrom_x'].astype(float)
##   d2['MMfrom_y']=d2['MMfrom_y'].astype(float)
##   d2[['MMto_x','MMto_y']] = d2.MMto.str.split(expand=True)
##   d2['MMto_x'] = d2['MMto_x'].str.strip(',')
##   d2['MMto_x']=d2['MMto_x'].astype(float)
##   d2['MMto_y']=d2['MMto_y'].astype(float)
   
   d2['edgeGeom'] = d2.edgeGeom.apply(lambda x: shapely.wkt.loads(x))
   d2['ListGeom'] = d2.edgeGeom.apply(lambda x:list(zip(x.xy[0].tolist(),x.xy[1].tolist())))
   d2['Frec_dist']=d2.apply(lambda x: frdist([(x.gps_x,x.gps_y)],x.edgeGeom),axis=1)
   d2['orth_dist']=d2.apply(lambda x: Odist((x.gps_x,x.gps_y),x.ListGeom),axis=1)
   d2['ErrorObs']=d2.apply(lambda x: 1 if LineString(x.ListGeom).distance(Point(x.gps_x,x.gps_y)) > 5 else 0,axis=1)
    
##   d2['orth_x']=d2.apply(lambda x:GetClosestPointx(x.MMfrom_x,x.MMfrom_y,x.MMto_x,x.MMto_y,x.gps_x,x.gps_y),axis=1)
##   d2['orth_y']=d2.apply(lambda x:GetClosestPointy(x.MMfrom_x,x.MMfrom_y,x.MMto_x,x.MMto_y,x.gps_x,x.gps_y),axis=1)
##   
   d2['Length_of_road_segment']=d2.apply(lambda x: x.edgeGeom.length,axis=1)
   #d2['Frechet_dist_per_Length']=d2['Frec_dist']/d2['Length_of_road_segment']
   d2['AreaPolygon'] = d2.apply(lambda x: AreaPloy([(x.gps_x,x.gps_y)] + x.ListGeom),axis=1)
   d2['SegSpeed'] = CalculateSegmentSpeed(d2)
   #d11=d2.groupby('edge').Frechet_dist_per_Length.agg(['count','min','max','mean'])
   #d10=d2.groupby('edge').Frec_dist.agg(['count','min','max','mean'])

   ##

   ###### Feature selction###
   features=['DistDiff','orth_dist']
   #features=['DistDiff','orth_dist','Frec_dist']
   #features=['DistDiff','orth_dist','AreaPolygon','Frec_dist']#'velocityFiltered','AreaPolygon','Frec_dist','orth_dist',]#
   
   NPx = d2[features].to_numpy() #returns a numpy array
   min_max_scaler = MinMaxScaler()
   NPx_scaled = min_max_scaler.fit_transform(NPx)
   dfF = pd.DataFrame(NPx_scaled,columns=features)
   
   
   #d2['observed'] = d2.TrMode.apply(lambda x: 0 if x == 'car' else 1)
   testY = d2.TrMode.apply(lambda x: 0 if x == 'car' else 1).values
   #Implementation of K-means
   d6= d2.copy()
   km = KMeans(n_clusters=2)
   y_predicted_KM = km.fit_predict(dfF)
   d6['cluster']=y_predicted_KM
   #PlotPointsAfterClustering(d6)   
   acKM.append(metrics.accuracy_score(testY,y_pred=y_predicted_KM))
   print(file)
   print('---------------- ------------ ')
   # Implementation of Gaussian Mixture

   d9=d2.copy()
   z=StandardScaler()
   #X[features]=z.fit_transform(X)
   EM=GaussianMixture(n_components=2)
   EM.fit(dfF)
   y_predicted_GMM=EM.predict(dfF)
   #X['cluster1']=cluster1
   #X['cluster1'].value_counts()
   
   #print(EM.means_)
   #### --- assign 0 to less errouneous points ### 
   cl1 = EM.means_[0]
   cl2 = EM.means_[1]
   Rcl1 = np.sum(np.square(cl1))
   Rcl2 = np.sum(np.square(cl2))
   if Rcl1 > Rcl2:
      y_predicted_GMM = 1 - y_predicted_GMM
      print('----> Rc1 > Rc2')
   ####
   d9['cluster']=y_predicted_GMM
   acGMM.append(metrics.accuracy_score(testY,y_pred=y_predicted_GMM))
   ####### added after rejection in Ictai 2021 #######
   edgecolourdf = d9[['cluster','edge']].groupby(['edge']).agg(pd.Series.mean)
   edgecolourdf.reset_index(inplace=True)
   d9['edgeClusterVoted'] = d9.edge.apply(lambda x: edgecolourdf[edgecolourdf.edge == x].cluster.values[0])

   
   d9['edgeColour'] = d9.edgeClusterVoted.apply(lambda x: 1 if x > 0.5 else 0)
#   if 1 == 1:
   df5G=d9[['u','v','edge','edgeColour']].drop_duplicates(subset = 'edge', keep = 'first')
   mgc = nx.from_pandas_edgelist(df5G, source='u', target='v', edge_attr=['edge','edgeColour'], create_using=nx.MultiGraph())
   d9['edgeColour'] = d9.apply(lambda x: CheckDegrees(x.u,x.v,mgc) if x.edgeColour == 1 else x.edgeColour,axis=1)


   df4G=d9[['u','v','edge','edgeGeom']].drop_duplicates(subset = 'edge', keep = 'first')
   mg = nx.from_pandas_edgelist(df4G, source='u', target='v', edge_attr='edge', create_using=nx.MultiGraph())

   disconEdges = FindDisconnectedSegments(mg)
   clawEdges = FindClaws(mg)
   #### red colour disconnected edges ###
   d9.loc[(d9.edge.isin(disconEdges)),'edgeColour'] = 1  
   d9.loc[(d9.edge.isin(clawEdges)),'edgeColour'] = 1 
   #d9[d9.edge.isin(disconEdges)].edgeColour
   #########################
   print(d9.loc[(d9.edge.isin(clawEdges)),'SegSpeed'].unique())
   PlotPointsAfterClustering(d9)
   d2['KM'] = y_predicted_KM
   d2['GMM'] = y_predicted_GMM
   #break
#if 1==1:
   countdf = d9[['edge','edgeColour']].drop_duplicates()
   totalSegments = len(countdf)
   redSegments = len(countdf[countdf.edgeColour == 1])
   dataset = int(file.replace('MM_geolife_DataSet_','').replace('.csv','')) #int(file[19:21])
   valdf = valdf.append({"traj":dataset,"totalSegment":totalSegments,"EstimatedError":redSegments},ignore_index=True)
   print('----end----')

   totalGNSSPoints =  totalGNSSPoints + len(df1)
   file_end = time.process_time()
   print(file_start - file_end)
   print('-------')

   
print(totalGNSSPoints)
valdf.astype(int).to_csv('Unspuervised_valdf_output.csv',header=True)
if 1 == 1:
   print(np.mean(acKM))
   print(np.mean(acGMM))
   lg1 = 'GMM'
   lg2 = 'KM'
   poix = list(range(0,len(acKM)))
   fig, ax = plt.subplots()
   ax.plot(poix,acGMM, c='red',marker='*', linestyle='-',label=lg1)
   ax.plot(poix,acKM,c='blue',marker='o', linestyle='-',label=lg2)
   #plt.xticks(np.arange(1, len(x)+1))
   ax.legend(loc='lower right', frameon=False)
   #plt.xticks(poix, xticks, rotation=90)
   plt.grid(True)
    #plt.ylim(0,1)
   plt.show()

#def Correct2DegRedEdges(u,v):

   



##   resdf = pd.DataFrame()
##
##   errorL = []
##   geomL = []
##   edgeL = []
##            #trimd9 = d9[d9.distance > 20]
##   for n,g in d9.groupby('edge'):
##      errorL.append(np.bincount(g.cluster.values).argmax())
##      geomL.append(g.edgeGeom.values[0])
##      edgeL.append(g.edge.values[0])
##      #print(error)
##   trimd9 = pd.DataFrame({'edge':edgeL,'edgeGeom':geomL,'cluster':errorL})   
##   if 1 == 1:
##      d = trimd9#[['edgeGeom','cluster','edge']].copy()
##      #d = d.drop_duplicates(subset=['cluster','edge'])
##      #d.loc[d.duplicated(subset=['edgeGeom','cluster','edge']), ['edgeGeom'] = np.nan
##      e = d.edgeGeom.shift(1).dropna().tolist()
##      c = d.cluster.shift(1).dropna().tolist()
##      e.append(Point(0,0))
##      c.append(0)
##      d['NedgeGeom'] = e
##      d['Ncluster'] = c
##      d['FedgeGeom'] = d.apply(lambda x: ops.linemerge(geometry.MultiLineString([x.edgeGeom,x.NedgeGeom])) if ((x.cluster == x.Ncluster) and (x.edgeGeom.coords[-1] == x.NedgeGeom.coords[0])) else x.edgeGeom,axis=1)
##
##   derrorL = d.cluster.to_list()
##   dgeomL = d.FedgeGeom.to_list()
##   print(len(derrorL))
##   print(len(derrorL) - np.bincount(derrorL)[0])
##   PlotPointsAfterClustering(d9,derrorL,dgeomL)
         #outname = 'Out_' + str(file)
      
      #d2.to_csv(outname,header=True)
      
##   # Implementation of DBSCAN
##   d21=d2.copy()
##   datDB=d21[['orth_dist','Frechet_dist_per_Length']]
##   datDB=datDB.values.astype('float32', copy=False)
##   dat_scaler=StandardScaler().fit(datDB)
##   dat=dat_scaler.transform(datDB)
##   model=DBSCAN(eps=0.25 ,min_samples=10 , metric='euclidean').fit(datDB)
##   outliers=dat[model.labels_ ==-1]
##   clusters=dat[model.labels_ !=-1]
##   d21['cluster']=model.labels_
   #PlotPointsAfterClustering(d21)

##colors=model.labels_
##colors_clusters=colors[colors!=-1]
##colors_outliers='black'
##clusters=Counter(model.labels_)
##print(clusters)
##print('Number of clusters= {}'.format(len(clusters)))
