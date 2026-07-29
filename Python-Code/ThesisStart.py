# -*- coding: utf-8 -*-
# Version vom 29. Juli 2026

import numpy as np
import matplotlib.pyplot as plt
import DataImportSensorBA as ds
import ThesisAnalysis as ta
import SpatialAnalysisBA as sa
import PlotBA as pt
from cartopy import crs


fLon_ref, fLat_ref = ds.GDictConfig.get ( "CentralLocation" )
CCRS_azmequi = crs.AzimuthalEquidistant ( central_longitude = fLon_ref, central_latitude = fLat_ref )

## typischer Ablauf für den Download einer Tages-Serie
"""
sURL = "https://archive.sensor.community/2026-02-07/"
ds.ListAllRefURL ( sURL = sURL, sExt = "csv", sSaveFilePath = "20260207_AllFiles.txt" )
"""


"""
ds.DownloadFiles ( sPlaceName = "Munich, Germany", 
                   sFileContainingAllLinks = "C:/DATA/Daten/Kriging/SensorCommunity/FileListings/20260206_AllFiles.txt", 
                  sDataType = "any", iStartLink = 0 ) 
"""

"""
sFolderPath = "C:/DATA/Daten/Kriging/SensorCommunity/Feb2026/20260205"
ds.SelectUsableFiles ( sFolder = sFolderPath, sCopyFolder = None, sDataType = "Matter" )
ds.SelectUsableFiles ( sFolder = sFolderPath, sCopyFolder = None, sDataType = "Temperature" )
"""

########## Abspeichern der Daten als *.npy Datei
"""
aDays = np.arange ( start = 1, stop = 31 + 1 )
sDataType = "Matter"
sSubFolderMonthYear = "Jul2025"#"Nov2025" ##"Apr2026" 
sMonth =  "07" ##"04"
sYear =  "2025" ##"2026" 
sDate = "2025-07-01" ## nur für Testzwecke
### WICHTIG: in der Datei DataImportSensor das Dict anpassen!
ds.SaveData ( tDaySequence = aDays, sMonth = sMonth, sYear = sYear, sDataType = sDataType, sSubFolderMonthYear = sSubFolderMonthYear, 
              bCheckAbnormality = True, bWriteFile = True, sLogFile = "SaveLog.txt"  )

### Check
aDateTime, aData, tInfo = ds.LoadData ( sSensorID = "45011", tSequence = None, sDate = sDate, sDataType = sDataType, 
                                        sSubFolderMonthYear = sSubFolderMonthYear )
ds.SaveDataStatisticDay ( sDataType = sDataType, sSubFolderMonthYear = sSubFolderMonthYear, bWriteFile = True )
### Check
aAverageDateTimeAll, aAverageDataAll, aID, DictInfo = ds.LoadDataStatisticDay ( sDataType = sDataType, sSubFolderMonthYear = sSubFolderMonthYear )

ds.SaveDataStatisticMonth ( sDataType = sDataType, sSubFolderMonthYear = sSubFolderMonthYear )
### Check
aRawData, DictStatistic = ds.LoadDataStatisticMonth ( sDataType = sDataType, sAverageParameter = "mean", sSubFolderMonthYear = sSubFolderMonthYear )

ds.SaveAllSensorLocations ( sSubFolderMonthYear = sSubFolderMonthYear )
"""

sDataSelection = "AzmP2"
sAverageParameter = "median"


sDate = "Nov. 2025"
sSubFolderMonthYear = "Nov2025"

ta.RunAnisotropicCrossValidation ( sModel = "matern", tParameterLambda = ( 1.0, 4.0, 4 ), tParameterVar = ( 2.0, 6.0, 3 ), 
                                   iNumStepsTheta = 18, sDate = sDate, sSubFolderMonthYear = sSubFolderMonthYear, 
                                   sDataSelection = sDataSelection, fVar_fix = 4.0, fLambda_fix = 3.0, sAverageParameter = sAverageParameter )

sDate = "Apr. 2026"
sSubFolderMonthYear = "Apr2026"

ta.RunAnisotropicCrossValidation ( sModel = "matern", tParameterLambda = ( 1.0, 4.0, 4 ), tParameterVar = ( 0.8, 1.2, 3 ), 
                                   iNumStepsTheta = 18, sDate = sDate, sSubFolderMonthYear = sSubFolderMonthYear, 
                                   sDataSelection = sDataSelection, fVar_fix = 1.0, fLambda_fix = 3.0, sAverageParameter = sAverageParameter )

sDate = "Jul. 2025"
sSubFolderMonthYear = "Jul2025"

ta.RunAnisotropicCrossValidation ( sModel = "matern", tParameterLambda = ( 1.0, 4.0, 4 ), tParameterVar = ( 0.6, 1.0, 3 ), 
                                   iNumStepsTheta = 18, sDate = sDate, sSubFolderMonthYear = sSubFolderMonthYear, 
                                   sDataSelection = sDataSelection, fVar_fix = 0.8, fLambda_fix = 3.0, sAverageParameter = sAverageParameter )



ta.ShowSensorNearLUA ( sKeyRefLUA = "Lan" )



ta.AnalyzeTimeAverageDataCV ( sDate = "November 2025", sSubFolderMonthYear = "Nov2025", sDataSelection = "AzmP2", 
                              tNumLags = ( 7, 20 ), tMaxLags = ( 4000, 22000, 50 ), tModels = ( "exponential", "spherical", "matern" ),
                              sAverageParameter = "median" )

ta.ShowDirectionalVariogram ( sDataSelection = "AzmP2", sAverageParameter = "median", sMonthYear = None, bShowFit = True )
    




