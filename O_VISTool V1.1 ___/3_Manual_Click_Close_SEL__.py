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

import pyperclip as pc
from selenium.webdriver.common.keys import Keys
import os
import sys
import time
import shutil
import keyboard
max_wait = 300


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

def PlotPointsAfterClustering(df1):
    bbox = df1.Bbox.to_list()[0:4]
    return None


######### remember the following for GeoLife data
###     ## utm.from_latlon(29.67698333, 116.0095667) =  (404164.7053251712, 3283403.500659356, 50, 'R')
chromeOptions = webdriver.ChromeOptions()
chromeOptions.add_argument('window-size=1920x1080')
chromedriver = os.getcwd() + '\\chromedriver.exe'
driver = webdriver.Chrome(executable_path=chromedriver,options=chromeOptions)
print(driver.title)
CheckDriver(driver)

clusterColour = ['g','r','b','k'] ## for 3 clusters
modeColour = {'car':'m','walk':'c'}

OutFilePath = "Results_FoliumMaps_MMfgeo_files"
filenames = listdir(OutFilePath)

foliumFiles = [ffile for ffile in filenames]
print('now')

time_per_df = pd.read_csv('time_per_df_MM_Vis.csv',index_col=0) ##columns=['dataset','MM_','MM_v','MM_e','MM_ae']) # 
Manual_click_df = pd.DataFrame(columns=['dataset','MM_','MM_v','MM_eL','MM_ec','MM_aeL','MM_aec'])
err_man_df = pd.DataFrame(columns=['dataset','no_error','error_seg','type_error'])

osmList = []
        
for i,file in enumerate(foliumFiles):
    #print(i)
    #c_dir = city_udf.loc[i,'direction'] 
    #c_siteid = city_udf.loc[i,'site_id'] 
    #file.split()
    file_name = file #str(i)+'_'+str(c_siteid) + '_' + str(c_dir) + "_FoVis.html"
    
    filepath=os.path.join(OutFilePath,file_name)
    print(filepath)
    mapFname = filepath  #'MM_Fo_vis1.html' ## we start from this html file
    
    mapUrl = 'file://{0}/{1}'.format(os.getcwd(), mapFname)       
 
    driver.get(mapUrl)
    click_counter = 0
    list_a = []
    _popup_xpath_str =  "//div[@class='leaflet-popup-content']"
    print("initiating.....")
    print("to Quit ..... Please press and hold SPACE, then close the element, then press 'q'  - - - ")
    print("Please press and hold SPACE, then click on the element, then press 's' to save - - - ")
   
    ## Check for how the clicks are happening 
    while True:#click_counter < Max_click_allowed: ## CheckDriver(driver)
        ## Regester your click event ##
        element = WebDriverWait(driver, 1000).until(EC.presence_of_element_located((By.XPATH, _popup_xpath_str )))
        _con = driver.find_element_by_xpath(_popup_xpath_str)  
        a = str(_con.text)
        
        ## Decision after a click event  ##
        if len(a) > 0 and keyboard.is_pressed('space') :

            keyboard.wait('s') #to save
            
            click_counter = click_counter + 1
            list_a.append(ast.literal_eval(a))
            osmList.append(list_a)
            print("You have saved the following string -> ")
            print(list_a)

        #### icon close operation  ## 
        element = WebDriverWait(driver, 1000).until_not(EC.presence_of_element_located((By.XPATH, _popup_xpath_str )))
        print('- icon closed')
        print("to Quit ..... Please press and hold SPACE, then close the element, then press 'q'  - - - ")
        print("Please press and hold SPACE, then click on the element, then press 's' to save - - - ")
        if keyboard.is_pressed('space'):
            print('you have to press q')
            keyboard.wait('q') ## to quit
            
            print('I am done with this file - ~ ~ ~ ')
            break
        else:
            continue
        ##
    ## Keep the while loops continue  ## 
        
    sys.stdout.flush()
    try:
        print('~ Saved in the file. ~')
        ##city_udf.loc[i,'e_osmid'] = ast.literal_eval(a)
        print(list_a)
        print(click_counter)## = click_counter + 1
    except:
        print(".........I can't save this object ->  Skipping! ")
        print(a)
    print('- - - -      - - -')
    print(osmList)
    print('   - - End of Loop - -      ')


print(osmList)


driver.quit()
#
#a.to_csv('_Final_Results.csv')
    #time.sleep(5)
    #done = page_wait(driver, By.ID, 'v-main', max_wait)    
    #driver.save_screenshot(png_fig_name_str)
    
