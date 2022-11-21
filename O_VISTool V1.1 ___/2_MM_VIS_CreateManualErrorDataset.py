###temp
from datetime import datetime,timedelta
from collections import Counter, defaultdict
import math
from math import exp, sqrt
from os import listdir
from os.path import isfile, join
import utm
import ast
import requests
import itertools
from itertools import permutations
from pyproj import Transformer,Proj, transform
import seaborn as sns

import pandas as pd
import geopandas as gpd
import numpy as np
from numpy.linalg import matrix_power
import osmnx as ox
import networkx as nx
import shapely
from shapely import wkt
from shapely.ops import nearest_points
from shapely.geometry import shape, Point, mapping, MultiPoint, LineString,box, Polygon
from shapely.ops import linemerge
from shapely.geometry import box, Polygon
from shapely.geometry.linestring import LineString

import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams.update({'font.size': 18})

# Order each plot based on each day,
# flow/hour as the indicator

from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from sklearn.metrics import mean_squared_error as ms

    
#### ------------ FIGURE_OSM PLOT ____

import folium
import webbrowser
from folium import plugins
from IPython.display import IFrame
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains


from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

from selenium.webdriver.remote.command import Command

from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


from selenium.webdriver.common.keys import Keys


import networkx as nx
import matplotlib.pyplot as plt

import pyperclip as pc
import os
import re
import sys
import shutil
import zipfile
import shutil
from datetime import datetime,timezone
import pathlib
import time
max_wait = 300
ox.config(use_cache=True, log_console=True)  

#curTime = DateTime.Now.ToString("yyyy-MM-dd hh:mm:ss tt") 

def add_edge_to_graph(G, e1, e2, w):
    G.add_edge(e1, e2, weight=w)
    
def using_concat(dfs):
    result = pd.concat(dfs, sort=False)
    n = result.index.nlevels
    return result.groupby(level=range(n)).first()

#%% Make a fucntion for checking if there's an active download


# Check that the browser is active
def get_status(driver):
    try:
        driver.execute(Command.STATUS)
        return "Alive"
    except (socket.error, httplib.CannotSendRequest):
        return "Dead"


# Wait timer for page to load
def page_wait(driver, class_to_wait_for, identifier_to_wait_for, max_wait):
    try:
        WebDriverWait(driver, max_wait).until(EC.presence_of_element_located((class_to_wait_for, identifier_to_wait_for)))
        done = True
    except TimeoutException:
        print('Landing page timeout')
        done = False
    return done

def CheckResponseClick(finding_String): ## check and click on a clicable element 
    WebDriverWait(driver, timeout=max_wait).until(EC.element_to_be_clickable((By.XPATH, finding_String)))
    while True:
        try:
            button_to_click = driver.find_element_by_xpath(finding_String)
            button_to_click.click()
            break
        except:
            disp_string = ' ! Waiting for response !  - > ' + str(finding_String)
            print(disp_string)
            time.sleep(5)
            
    return button_to_click 

def CheckResponseSEND_KEYS(finding_String): ## wait and send_keys
    WebDriverWait(driver, timeout=max_wait).until(EC.element_to_be_clickable((By.XPATH, finding_String)))
    while True:
        try:
            field_to_send = driver.find_element_by_xpath(finding_String)
            break
        except:
            disp_string = ' ! Waiting to Communicate !!! ' + str(finding_String)
            print(disp_string)
            time.sleep(5)
            
    return field_to_send

def CheckDriver(driver):
# Check that the browser is active
    if get_status(driver) == 'Alive':
        print('Browser active\n')
    else:
        print('An error occurred\n')
        sys.stdout.flush()
        quit()
    return

def Create_FoliumHTML(Point_location,Point_popup,Point_div_icon,Point_popup_str,filepath):                                    
    aGnp = ox.graph_from_point(Point_location, dist=50, retain_all='True', simplify=True,truncate_by_edge=True)
    aGEnp = ox.graph_to_gdfs(aGnp, nodes=True)[1]
    MM_gdf_nodes = ox.graph_to_gdfs(aGnp, nodes=True)[0]
    #aGEnp.reset_index(inplace=True)
    #Create osmid enabled graph ## osmid edge id                  
    mm_fo = ox.plot_graph_folium(aGnp, popup_attribute="osmid", weight=4, color="blue",zoom=5)#,tiles='openstreetmap'
##    folium.Marker(location=Point_location,3,
##              popup=Point_popup
##              ,icon=folium.DivIcon(Point_div_icon)
##           ).add_to(mm_fo)
##    folium.CircleMarker(Point_location ,4,fill=True,color='red').add_child(folium.Popup(Point_popup_str)).add_to(mm_fo)
    folium.Marker(location=Point_location, popup=Point_location, tooltip=Point_popup, icon=folium.DivIcon(Point_div_icon), draggable=True).add_to(mm_fo)
    mm_fo
    mm_fo.save(filepath)

ox.config(use_cache=True, log_console=True)
ox.config(use_cache=True, log_console=True) 




path = "MM_Trajectories_beforeRevision"
OutFilePath = 'Results_FoliumMaps_MMfgeo_files/'
filenames = listdir(path)

EligibleFiles = [ffile for ffile in filenames]
print('now')
totalGNSSPoints = 0

time_per_df = pd.read_csv('time_per_df_MM_only.csv',index_col=0) #columns=['dataset','MM_','MM_v','MM_e','MM_ae']

for i_row,file in enumerate(EligibleFiles):
    file_start = time.process_time()

    time_per_df.loc[i_row,'dataset'] = file
    
    fPath = str(path) + "/" + str(file)
    rawdf = pd.read_csv(fPath)
    ###########
    rawdf = rawdf.reset_index(drop=True)
    rawdf['point'] = rawdf.point.apply(lambda x: ast.literal_eval(x))
    rawdf['MMfrom'] = rawdf.MMfrom.apply(lambda x: ast.literal_eval(x))
    rawdf['MMto'] = rawdf.MMto.apply(lambda x: ast.literal_eval(x))
    
    rawdf['edgeGeom'] = rawdf.edgeGeom.apply(lambda x: shapely.wkt.loads(x))
    rawdf['ListGeom'] = rawdf.edgeGeom.apply(lambda x:list(zip(x.xy[0].tolist(),(x.xy[1].tolist()))))
    rawdf['MM_Edge_raw']= rawdf.ListGeom.apply(lambda x: [utm.to_latlon(i[0],i[1], 50, 'N') for i in x])
    rawdf['MM_Edge_geom']= rawdf.MM_Edge_raw.apply(lambda x: LineString(x))
                    
    rawdf['PPoint'] = rawdf.point.apply(lambda x: Point(x))
                                             
    rawdf['Point_raw'] = rawdf.PPoint.apply(lambda x: utm.to_latlon(x.x,x.y, 50, 'N') ) ## this converts projected point to lat lon ## 
    rawdf['Point_geom'] = rawdf.Point_raw.apply(lambda x: Point(x) )

    ### process trajectory ## get the list of lat-lon of map-matched route ##
    
    #rawdf['MMpoint_raw'] = rawdf.point.apply(lambda x: ast.literal_eval(x))
    #rawdf['MMpoint_Geom'] =
    

    df1 = rawdf.copy()
    
    Raw_df = gpd.GeoDataFrame(df1[['Point_raw','edge', 'Point_geom', 'edgeGeom']], geometry = "Point_geom") ## china :32650
    Raw_df['P_lat'] = Raw_df.Point_raw.apply(lambda x: x[0])
    Raw_df['P_lon'] = Raw_df.Point_raw.apply(lambda x: x[1])
    
    Edge_df = gpd.GeoDataFrame(df1[['Point_raw','edge', 'MM_Edge_raw', 'MM_Edge_geom','u', 'v','trajectory','MMfrom', 'MMto']], geometry = "MM_Edge_geom")
    Edge_df.drop_duplicates(['edge'],inplace=True)
    Edge_df.rename({'edge':'key'},axis=1,inplace=True)
    Edge_df.MMfrom = Edge_df.MMfrom.apply(lambda x: utm.to_latlon(x[0],x[1], 50, 'N') )
    Edge_df.MMto = Edge_df.MMto.apply(lambda x: utm.to_latlon(x[0],x[1], 50, 'N') )
    
    Edge_df['FP_lat'] = Edge_df.Point_raw.apply(lambda x: x[0])
    Edge_df['FP_lon'] = Edge_df.Point_raw.apply(lambda x: x[1])
    
    MM_gdf_edges = Edge_df.set_index(['u', 'v', 'key'])
    
 
    
    ##aGnp = ox.graph_from_point(gPoint, dist=400, network_type='drive', simplify=True)
    Point_Lat_lon_list = Raw_df.Point_raw.to_list() ## list(zip(gLat,gLon))
                    
    bbox = df1.Bbox.to_list()[0:4]
     #Create the Graph from BBox
    aGnp= ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3],truncate_by_edge=True)#,network_type="drive")
    aGEnp = ox.graph_to_gdfs(aGnp, nodes=True)[1]
    MM_gdf_nodes = ox.graph_to_gdfs(aGnp, nodes=True)[0]
    #aGEnp.reset_index(inplace=True)
    
     #Create osmid enabled graph ## osmid edge id                  
    mm_fo = ox.plot_graph_folium(aGnp, popup_attribute="osmid", weight=2, color="grey")#,tiles='openstreetmap'
    #Plot the raw lat lon points
    data_P = Raw_df
    for i in range(0,len(data_P)):
##       folium.Marker(
##          location=[data_P.iloc[i]['P_lat'], data_P.iloc[i]['P_lon']],
##          popup=data_P.iloc[i]['edge']
##          ,icon=folium.DivIcon(html=f"""<div style="font-family: courier new; color: red">{data_P.iloc[i]['edge']}</div>""")
##       ).add_to(mm_fo)
       
       folium.CircleMarker([data_P.iloc[i]['P_lat'], data_P.iloc[i]['P_lon']],3,fill=True,color='red').add_child(folium.Popup(str(data_P.iloc[i]['edge']))).add_to(mm_fo)
    ##folium.PolyLine(Point_Lat_lon_list, color="purple", weight=2, opacity=1,dash_array='10').add_to(mm_fo)

    ### Now plot the map-matched route
    ## try popopping up the osmid ##
    ## Use the osmid of the map-matched route

    MM_traj_lat_lon  = list(Edge_df[['MMfrom','MMto']].values.flatten())
    folium.PolyLine(MM_traj_lat_lon, color="blue", weight=2, opacity=1,dash_array='10').add_to(mm_fo)

    ### iterative polyline with popup_
    for i,r in Edge_df.iterrows():
        folium.Marker(location=r['MMfrom'], popup = r['key'],tooltip='MM from').add_to(mm_fo)
        folium.Marker(location=r['MMto'], popup = r['key'],tooltip='MM To').add_to(mm_fo)
        
    ##MM_graph = ox.graph_from_gdfs(MM_gdf_nodes, MM_gdf_edges)
    #for i,r in Edge_df.iterrows():
    #    folium.Marker(location=r['Point_raw'], popup = r['key']).add_to(mm_fo)
##    ## you can add points in the loop and add markers!
##    data = Edge_df
##    for i in range(0,len(data)):
##       folium.Marker(
##          location=[data.iloc[i]['P_lat'], data.iloc[i]['P_lon']],
##          popup=data.iloc[i]['key']
##         ,icon=folium.DivIcon(html=f"""<div style="font-family: courier new; color: blue">{data.iloc[i]['key']}</div>""")
##       ).add_to(mm_fo)
##    
    ###
    ##create a graph of these points as unique Edge id ## display the edge id in the popup_attribute ##
    ##MM_graph = ox.graph_from_gdfs(MM_gdf_nodes, MM_gdf_edges)
    ## creating MM route from MMfrom and MMto columns
    ##mm_route = list(Edge_df[['MMfrom','MMto']].values.flatten())
    ##mm_fo = ox.plot_graph_folium(MM_graph, graph_map=mm_fo, popup_attribute="key", weight=7)
    ##mm_route = list(Edge_df[['u','v']].values.flatten())
    ##mm_fo = ox.plot_graph_folium(aGnp, popup_attribute="osmid", weight=2, color="grey",tiles='openstreetmap')
    

    ### next plot the identified error plot

    mm_fo
    out_filepath =  OutFilePath + str(file.split('.')[0]) + "_FoVis.html"
    
    mm_fo.save(out_filepath)#,png_enabled=True)

    file_end = time.process_time()
    process_time = file_start - file_end

    time_per_df.loc[i_row,'MM_v'] = process_time ## insert file name
    print('-------')

End_time = time.process_time()
#print(End_time - start)
time_per_df.to_csv('time_per_df_MM_Vis.csv')

##
###### Read city files one by one---
##if True:
##    City_path = "MapMatcched Trajectories Geolife/" ###according to the date
##    City_files_list = os.listdir(City_path)
##    City_df_list = []
##    City_h = ['site_id', 'site_name', 'lat', 'lon', 'full_date', 'date', 'month',
##       'year', 'dayofweek', 'duration', 'Direction', 'count']
##
##    for cur_file in City_files_list:       
##        print(cur_file)
##        cur_filepath = os.path.join(City_path,cur_file)
##        cur_file_df = pd.read_csv(cur_filepath, index_col=None)
##        City_df_list.append(cur_file_df)
##
##    City_df = pd.concat(City_df_list,axis=0,ignore_index=True)
##
##city_udf = pd.DataFrame(columns=['site_id','site_name','lat','lon','direction','e_osmid'])
##gb = City_df.groupby(['site_id','site_name','lat','lon','direction'])
##
##
##for i,tdf in enumerate(gb):
##    #print(i)
##    print(len(tdf[1]))
##    city_udf.loc[i,'site_id'] = tdf[0][0]
##    city_udf.loc[i,'site_name'] = tdf[0][1]
##    city_udf.loc[i,'lat'] = tdf[0][2]
##    city_udf.loc[i,'lon'] = tdf[0][3]
##    city_udf.loc[i,'direction'] = tdf[0][4]
##
##    Point_site_name = tdf[0][1]
##    Point_direction = tdf[0][4]
##    
##    Point_location = [tdf[0][2], tdf[0][3]]
##    Point_popup = Point_site_name
##    
##    Point_div_icon = html=f"""<div style="font-family: courier new;font-weight: bold; color: red">{Point_direction}</div>"""
##    Point_popup_str  = str(Point_site_name)
##    
##                                    
##    file_name = str(i)+'_'+str(tdf[0][0]) + '_' + str(tdf[0][4]) + "_FoVis.html"
##    
##    filepath=os.path.join(OutFilePath,file_name)
##    print(filepath)
##    Create_FoliumHTML(Point_location,Point_popup,Point_div_icon,Point_popup_str,filepath)
##    
##    ### create 150 points in the folium
##city_udf.fillna(0).to_csv('city_udf.csv')
##
##data_P = city_udf.copy()
