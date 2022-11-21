# Author:   Subhrasankha Dey


from shapely.geometry import shape, Point, mapping
from shapely.ops import linemerge
import fiona
import networkx as nx
import csv
from rtree import index
from math import exp, sqrt
from datetime import datetime
import itertools
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

startTime = datetime.now()
FMT = '%H:%M:%S'
optdf = pd.DataFrame(columns=['edge','time','point','edgeGeom'])

def mapMatch(points,TimeList, segmentInfo, graph, tree, segmentShapes, osmN,osmE,
             decayconstantNet, decayConstantEu, maxDist):
    
    # Ensure that distance parameters are floats
    decayconstantNet = float(decayconstantNet)
    decayConstantEu = float(decayConstantEu)
    maxDist= float(maxDist)

    # Probability distribution (based on Viterbi algorithm) of street
    # segments for a given GPS track point, with the most probable predecessor
    #  segment taken into account
    V = [{}]

    endpoints = segmentInfo[0]  # endpoint coordinates of all street segments
    lengths = segmentInfo[1]    # length of all street segments
    pathnodes = []              # set of path nodes to prevent loops
    TimeLine = []
    print("Matching Started-->")
    # Get segment candidates for the first point of the track
    for f in range(0,len(points)):
        
        sc = getSegmentCandidates(points[f], tree, segmentShapes, decayConstantEu,
                              maxDist)
        #print(sc)
        if sc:
            print('Start Point ->')
            break
    
    for s in sc:
        V[0][s] = {"prob": sc[s], "prev": None, "path": [], "pathnodes": [], "time":TimeList[f],"point":points[f]} ### Changed here on 17-November-2019 TimeList

    # Run Viterbi algorithm when t > 0
    Vc = 0 ## initiate Vcount, a counter that stores the prob of relevant points within a gps error circle
    tc = f
    lastsc = sc
    for t in range(f+1, len(points)):
        #print(t)
        # Store previous segment candidates
        # Get segment candidates and their a priori probability
        sc = getSegmentCandidates(points[t], tree, segmentShapes,
                                  decayConstantEu, maxDist)
        if sc:
            #print("SC empty, breaking here")
            #tc = tc + 1
            Vc = Vc + 1
            V.append({})
            for s in sc:
                max_tr_prob = 0     # init maximum transition probability
                prev_ss = None
                path = []
                for prev_s in lastsc: ## Returns index of edges. 
                    # determine the highest transition probability from previous
                    # candidates to s and get the corresponding network path
                    
                    pathnodes = V[Vc-1][prev_s]["pathnodes"][-10:]
                    n = getNetworkTransP(prev_s, s, graph, endpoints,osmN,osmE,
                                         lengths, pathnodes, decayconstantNet)
                    
                    np = n[0]   # network transition probability
                    tr_prob = V[Vc-1][prev_s]["prob"]*np
                    # Select the most probable predecessor candidate and the
                    # path to it
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_ss = prev_s
                        path = n[1]
                        if n[2] is not None:
                            pathnodes.append(n[2])
                # Final probability of a candidate is the product of a-priori
                # and network transitional probability
                
                max_prob = sc[s] * max_tr_prob
                V[Vc][s] = {"prob": max_prob, "prev": prev_ss, "path": path,
                           "pathnodes": pathnodes,"time":TimeList[t],"point":points[t]}

            lastsc = sc
            tc = t
            maxv = max(value["prob"] for value in V[Vc].values())
            maxv = (1 if maxv == 0 else maxv)
            for s in V[Vc].keys():
                V[Vc][s]["prob"] = V[Vc][s]["prob"]/maxv

    

    
    optdf = pd.DataFrame(columns=['edge','time','point','edgeGeom'])
    
    if not V[-1].values():
        return []
    else:
        max_prob = max(value["prob"] for value in V[-1].values()) # Get the highest probability at the end of the track
        previous = None

        if max_prob == 0:
            print(" probabilities fall to zero (network distances in data are too large, try increasing network decay parameter) ") 
            return []

        else:
        # Get the most probable ending state and its backtrack
            for st, data in V[-1].items():
                if data["prob"] == max_prob:
                    previous = st
                    temp_d = {'edge': st,'time': [data["time"]],'point':data["point"],'edgeGeom':osmE.geometry[st]}
                    temp_con_df = pd.DataFrame(pd.Series(temp_d)).T
                    optdf = pd.concat([optdf,temp_con_df],axis=0,ignore_index=True)  ## changed on 28-09-2022
                    break
            #print(V)
            #print(TimeList)
            # Follow the backtrack till the first observation to fish out most
            # probable states and corresponding paths
            for t in range(len(V) - 2, -1, -1):
                # Get the subpath between last and most probable previous segment and
                # add it to the resulting path
                path = V[t + 1][previous]["path"]
                key = next(iter(V[t]))                        ########## CHANGED HERE ON 19th_NOVEMBER-2019 ___> by S DEY -> Very important for keeping time stamped
                tim = V[t][key]["time"]
                #if len(path)> 0:
                    #TimeLine = len(path)*[tim] + TimeLine  #else [] ## insert
                pedge = V[t + 1][previous]["prev"]
                
                temp_d = {'edge': pedge,'time': [tim],'point':V[t + 1][previous]["point"],'edgeGeom':osmE.geometry[pedge]}
                temp_con_df = pd.DataFrame(pd.Series(temp_d)).T
                optdf = pd.concat([optdf,temp_con_df],axis=0,ignore_index=True)  ## changed on 28-09-2022
                previous = V[t + 1][previous]["prev"]

            optdf_ = optdf.iloc[::-1].copy() ### Reverse the order ## changed on 28-09-2022
            
            #optdf = optdf.groupby([optdf.edge],sort=False).time.apply(sum) ## add time lists
            
            print("---> MapMatching with Time-Stamp Worked ** Modified ** Important:) <----- Created by S Dey on 19-November-2019")
            return optdf_




# Return closest segment candidates with a-priori probabilities based on
# maximal spatial distance of segments from point
def getSegmentCandidates(point, tree, segmentShapes, decayConstantEu,
                         maxdist=20):
    candidates = {}
    make_point = Point(point)
    point_buffer = make_point.buffer(maxdist)
    for f_id in list(tree.intersection(point_buffer.bounds)):
        line = segmentShapes.get(f_id) ## is a DICTIONARY with edge id and segment geom 
        
        dist = make_point.distance(line)
        ##
        C = np.array(make_point.coords[0])
        A = np.array(line.coords[0])
        B = np.array(line.coords[1])
        odist = np.linalg.norm(np.cross(B - A, C - A))/np.linalg.norm(B - A)
        ##
        #print(odist)
        #print(dist)                            ## calculated orthogonal distance here ##        _> changed on 5th December, 2019 by S Dey
        if dist <= maxdist:
            candidates[f_id] = getPDProbability(dist, decayConstantEu)
    return candidates


# Return transition probability of going from segment s1 to s2
# takes two edge indexes
# and return the probability from one to the other
# changed and developed by S Dey -> 13-09-2019
def getNetworkTransP(s1, s2, graph, endpoints, osmN,osmE, segmentlengths, pathnodes,
                     decayconstantNet):
    subpath = []
    s1_point = None
    s2_point = None

    if s1 == s2:
        dist = 0
    else:
        # Obtain edges (tuples of endpoints) for segment identifiers
        
        s1_edge = endpoints[s1]
        s2_edge = endpoints[s2]

        # Determines segment endpoints of the two segments that are
        # closest to each other
        minpair = [0, 0, 100000]
        for i in range(0, 2):
            for j in range(0, 2):
                d = round(pointdistance(s1_edge[i], s2_edge[j]), 2)
                if d < minpair[2]:
                    minpair = [i, j, d]
        s1_point = s1_edge[minpair[0]]
        s2_point = s2_edge[minpair[1]]
        s1_osm = osmN[(osmN['x'] == s1_point[0]) & (osmN['y'] == s1_point[1])].osmid.values[0] ## find the right osmid from osmN table
        s2_osm = osmN[(osmN['x'] == s2_point[0]) & (osmN['y'] == s2_point[1])].osmid.values[0]
        if s1_point == s2_point:
            # If segments are touching, use a small network distance
            dist = 5
        else:
            try:
                # Compute the shortest path based on segment length on
                # street network graph where segment endpoints are nodes and
                #  segments are (undirected) edges
                if graph.has_node(s1_osm) and graph.has_node(s2_osm):
                    #dist = nx.astar_path_length(graph, s1_point, s2_point,weight='length',heuristic=pointdistance)
                    #path = nx.astar_path(graph, s1_point, s2_point,weight='length',heuristic=pointdistance)
                    path = nx.shortest_path(graph,s1_osm,s2_osm,weight='length') #nx.shortest_path_length(G, row['Orgin_nodes'], row['Destination_nodes'], weight='length'
                    dist = nx.shortest_path_length(graph,s1_osm,s2_osm,weight='length')
                    path_edges = zip(path, path[1:])
                    subpath = []
                    for e in path_edges:
                        #oid = graph.edge[e[0]][e[1]]["OBJECTID"]#
                        oid = osmE[(osmE.u == e[0])&(osmE.v== e[1])].index.values[0] 
                        subpath.append(oid)
                else:
                    dist = 3*decayconstantNet
            except nx.NetworkXNoPath:
                dist = 3*decayconstantNet
    return (getNDProbability(dist, decayconstantNet), subpath, s2_point)


# Return the probability that a point is on a segment
# dist: Euclidean distance between the point and the segment (in meter)
# decayconstant: Euclidean distance in meter farther than which probability
# falls to 0.34
def getPDProbability(dist, decayconstant=10):
    decayconstant = float(decayconstant)
    dist= float(dist)
    try:
        p = 1 if dist == 0 else round(1/exp(dist/decayconstant), 4)
    except OverflowError:
        p = round(1/float('inf'), 2)
    return p


# Return the probability that a segment is the successor of another on a track
# dist: Euclidean distance between two segments (in meter)
# decayconstant: Network distance in meter farther than which probability
# falls to 0.34
def getNDProbability(dist, decayconstant=30):
    decayconstant = float(decayconstant)
    dist = float(dist)
    try:
        p = 1 if dist == 0 else round(1/exp(dist/decayconstant), 2)
    except OverflowError:
        p = round(1/float('inf'), 2)
    return p


# Build an RTREE spatial index of the street network
# shapefile: a shapefile containing street segments (must be planarized)
def buildRTree(osmE):
    idx = index.Index()
    dfE = osmE.geometry.to_dict()
    streets = list(dfE.keys())
    #with fiona.open(shapefile, 'r') as streets:
    for st in streets:
        #st_id = int(st['properties']['OBJECTID'])
        st_geom = shape(dfE[st])#['geometry'])
        idx.insert(st, st_geom.bounds)
    return idx


# Build a network graph of the street
# shapefile: a shapefile containing street segments (must be planarized)
# segmentlengths: obtained from getSegmentInfo
def getNetworkGraph(og, segmentlengths):
    #g = og
    sg = og#.to_undirected()
    for idx, n in enumerate(list(sg.edges())):
        oid = int(sg[n[0]][n[1]][idx])
        sg[n[0]][n[1]]['length'] == segmentlengths[oid]
    return sg


# Returns a dictionary containing FID key and binary geom value
# shapefile: a shapefile containing street segments (must be planarized)
def getSegmentShapes(osmE):
    shapes = {}
    dfE = osmE.geometry.to_dict()
    streets = list(dfE.keys())
    for st in streets:
            #st_id = osmE.osmid.values[st]# 
        st_geom = shape(dfE[st])
        shapes[st] = st_geom#shapes[st_id] = st_geom
    return shapes


# Returns endpoints coordinates of all segments and their length
# shapefile: a shapefile containing street segments (must be planarized)
def getSegmentInfo(osmN,osmE):
    endpoints = []
    #endpoints.append({})
    pointPairs = list(zip(osmE.u,osmE.v))
    segmentlength = osmE.length.values #* 100000
    for po in pointPairs:
        O1 = po[0]
        iO1 = osmN.loc[osmN["osmid"] == float(O1)]
        to = (iO1.x.values[0],iO1.y.values[0])
        
        O2 = po[1]
        iO2 = osmN.loc[osmN["osmid"] == float(O2)]
        frm = (iO2.x.values[0],iO2.y.values[0])
        endpoints.append((to,frm))
    return endpoints, segmentlength



# Return distance between two points p1 and p2
# p1 or p2 must be a tuple of x and y coordinates
def pointdistance(p1, p2):
    dist = sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return dist

def OSMnodeDistance(O1,O2,osmN):
   iO1 = osmN.index[osmN["osmid"] == float(O1)].tolist()[0] #iO1 = osmN.index[osmN['OsmID'] == float(Trajectory[0])].tolist()[0]
   iO2 = osmN.index[osmN["osmid"] == float(O2)].tolist()[0]
   lat1 = (osmN.y[iO1]) * (math.pi/180)
   lat2 = (osmN.y[iO2]) * (math.pi/180)
   dlon = (osmN.x[iO1]- osmN.x[iO2])  * (math.pi/180)
   dlat = (osmN.y[iO1] - osmN.y[iO2])  * (math.pi/180)
   a = (math.sin(dlat/2))**2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon/2))**2 
   c = 2 * math.atan2( np.sqrt(a), np.sqrt(1-a) ) 
   d = 6373 * c 
   Dist = d
   return Dist * 1000
