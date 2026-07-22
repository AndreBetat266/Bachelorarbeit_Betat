# -*- coding: utf-8 -*-
# Version vom 27. Juni 2026

import numpy as np
import os
import glob
import shutil
from termcolor import colored
from Statistic import AnalyzeDataStructure, ScreenDataSeriesAbnormality
from Utilities import CheckAssert, SearchFilesInFolder, GetMonth
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import shapely
import tqdm
from GeodataAnalysis import ShowBorder 
from cartopy import crs


GDictConfig = {
    "DefaultSaveFolder"         : "C:/DATA/Daten/Kriging/SensorCommunity/Temp/",
    "DownloadArchiveURL"        : "https://archive.sensor.community/",
    #"BaseFolderDay"             : "C:/DATA/Daten/Kriging/SensorCommunity/Apr2026/",
    #"BaseFolderDay"             : "C:/DATA/Daten/Kriging/SensorCommunity/Nov2025/",
    "BaseFolderDay"             : "C:/DATA/Daten/Kriging/SensorCommunity/Jul2025/",
    "SaveFolder"                : "C:/DATA/Daten/Kriging/SensorCommunity/",
    "BaseFolderMonthLUA"        : "C:/DATA/Daten/Kriging/Landesumweltamt Bayern/",
    "HeaderKeywordsMatter"      : ( "P0", "P1", "P2", "P4" ),
    "HeaderKeywordTemperature"  : "temperature",
    "CentralLocation"           : ( 11.575328, 48.137371 ), #LonLat: 11.575328, 48.137371 #München Marienplatz
    "MinimalNumDifferentValues" : 20, 
    "MaxAllowedInterruption"    : 180.0,
    "ThresholdExcessParameter"  : ( 100.0, 20, 120 ) ## Schwellwert, Mindestlänge der Sequenzen, maximale Länge aller Sequenzen oberhalb Mindestlänge
    }
### Definiert welche Daten aus den CSV Dateien importiert werden für Messdaten P1 und P2
GDictMatterColumns = {
    "IndicesP1"        : ( ( 3, 4, 5, 6 ), 12, "$\mathrm{P}_{10}$" ),  #SDS011 > lat;lon;timestamp;P1;P2
    "IndicesP2"        : ( ( 3, 4, 5, 9 ), 12,  "$\mathrm{P}_{2.5}$"),  #SDS011 > lat;lon;timestamp;P1;P2
    "IndicesP1P2"      : ( ( 3, 4, 5, 6, 9 ), 12, ( "$\mathrm{P}_{10}$", "$\mathrm{P}_{2.5}$" ) ) #SDS011 > lat;lon;timestamp;P1;P2
    }
### Definiert welche Daten aus den CSV Dateien importiert werden für Messdatum Temperatur
GDictTemperatureColumns = {
    "BME280"           : ( ( 3, 4, 5, 9 ), 11, "$\mathrm{T}$" ), #lat;lon;timestamp;temperature
    "BMP280"           : ( ( 3, 4, 5, 9 ), 10, "$\mathrm{T}$" ),# lat;lon;timestamp;pressure;temperature
    "BMP180"           : ( ( 3, 4, 5, 9 ), 10, "$\mathrm{T}$" ), #lat;lon;timestamp:temperature
    "DHT22"            : ( ( 3, 4, 5, 6 ), 8,  "$\mathrm{T}$"  ), #lat;lon;timestamp;temperature;humidity
    "SHT31"            : ( ( 3, 4, 5, 6 ), 8,  "$\mathrm{T}$"  ) #lat;lon;timestamp;temperature;humidity
    }

# ********************************* Zusammenstellen der Daten für einen Zeitpunkt plus-minus delta Minuten *********************************
def LoadDataTimeFrame ( sDate, sStartTime, sDataType, iWindowDelta = 5, sSubFolderMonthYear = "Apr2026", bUseMeanMatches = False ):
    sDateFormat = "%Y-%m-%dT%H:%M"
    sDateTimeStart = sDate + "T" + sStartTime
    CDateTimeStart = datetime.strptime ( sDateTimeStart, sDateFormat )
    CDateTimeEnd = CDateTimeStart + timedelta ( minutes = iWindowDelta ) 
    print ( ">> LoadDataTimeFrame > Period from %s to %s" % ( CDateTimeStart, CDateTimeEnd ) )
    
    aDateTimeAll, aDataAll, aID_All, DictInfo, _ = LoadRawData ( sDataType = sDataType, sDate = None, sSubFolderMonthYear = sSubFolderMonthYear )
    ListData = list ()
    ListDataID = list ()
    iCounter = 0
    for sSensorID in DictInfo.keys ():
        iCountID, fLon, fLat, fLon_ref, fLat_ref = DictInfo.get ( sSensorID )
        
        aSelect = ( aID_All == iCountID )
        aDataID = aDataAll[ aSelect ] # CountID, P1, P2
        aDateTimeID = aDateTimeAll[ aSelect ]
        aSelect = np.logical_and ( aDateTimeID >= CDateTimeStart, aDateTimeID <= CDateTimeEnd )
        iNumMatches = np.sum ( aSelect )
        if ( iNumMatches >= 1 ):
            if ( bUseMeanMatches == True ):
                aDataMatch = np.mean ( aDataID[ aSelect ], axis = 0 )
            else:
                aDataMatch = aDataID[ aSelect ][ iNumMatches // 2 ]
            if ( aDataMatch.shape[ 0 ] >= 2 ):
                ## Aufpassen ! Die Reihenfolge von Lat und Lon ist andersherum als in der DictInfo hinterlegt
                ListData.append ( ( fLat, fLon, fLat_ref, fLon_ref, aDataMatch[ 0 ], aDataMatch[ 1 ] ) )
            else:
                ListData.append ( ( fLat, fLon, fLat_ref, fLon_ref, aDataMatch[ 0 ] ) )
            ListDataID.append ( iCountID )
            iCounter += 1
        else:
            print ( ">> LoadDataTimeFrame > No valid Data found for Sensor %s [%d]" % ( sSensorID, iCountID ) )
    
    aData = np.asarray ( ListData, dtype = np.float64 )
    aDataID = np.asarray ( ListDataID, dtype = np.int16 )
    print ( ">> LoadDataTimeFrame > Found %d valid Data Files, Data Shape %s" % ( iCounter, str ( aData.shape ) ) ) 
    
    return ( aData, aDataID, DictInfo )
# ************** Funktion, die die über LoadDataTimeFrame ermittelten Werte als Liste ausgibt, um diese händisch zu überprüfen *************
#               die Zusammstellung scheint wirklich wie gewünscht zu funktionieren; 13.06.2026 
def BuildControlList ( sDate, sStartTime, aRawData, aDataID, DictInfo ):
    ListCheck = list ()

    DictInfo_inv = { tTupel[ 0 ]: sKey for sKey, tTupel in DictInfo.items () }
    sDateTimeStart = sDate + "T" + sStartTime

    for ik, iID in enumerate ( aDataID ):
        sSensorID = DictInfo_inv.get ( iID )
        fP1, fP2 = aRawData[ ik,  -2 :  ]
        ListCheck.append ( ( sDateTimeStart, sSensorID, fP1, fP2 ) )


    print ( ListCheck )
    
    return ( ListCheck )

### möglicher Aufruf
### BuildControlList ( sDate, sStartTime, aRawData, aDataID, DictInfo )
# ****************************************** Einlesen der auf Festplatte abgelegten Daten **************************************************
## tSequence = ( sStartDatum, sEndDatum ) oder sDate müssen übergeben werden
def LoadData ( sSensorID, tSequence = None, sDate = None, sDataType = "matter", sSubFolderMonthYear = "Apr2026"):
    CheckAssert ( bBool = ( ( tSequence is not None ) or ( sDate is not None ) ), sMsg = "Either <tSequence> or <sDate> must be given!" )        
    sDateFormat = "%Y-%m-%dT%H:%M"
    if ( tSequence is not None ):
        CheckAssert ( bBool = ( len ( tSequence ) == 2 ), sMsg = "Invalid Format <tSequence>!" )
        sDateTimeStart = tSequence[ 0 ] + "T" + "00:00"
        CDateTimeStart = datetime.strptime ( sDateTimeStart, sDateFormat )
        sDateTimeEnd = tSequence[ 1 ] + "T" + "00:00"
        CDateTimeEnd =  datetime.strptime ( sDateTimeEnd, sDateFormat ) + timedelta ( days = 1 ) 
    else:
        sDateTimeStart = sDate + "T" + "00:00"
        CDateTimeStart = datetime.strptime ( sDateTimeStart, sDateFormat )
        CDateTimeEnd = CDateTimeStart + timedelta ( days = 1 ) 
    print ( ">> LoadData > Sensor %s; Period: %s to %s" % ( sSensorID, CDateTimeStart, CDateTimeEnd ) )
    
    aDateTimeAll, aDataAll, aID_All, DictInfo, DictDays = LoadRawData ( sDataType = sDataType, sDate = None,  
                                                                        sSubFolderMonthYear = sSubFolderMonthYear, bShowInfo = False )
    tInfo = DictInfo.get ( sSensorID )
    tDaysAvailable = DictDays.get ( sSensorID )
    
    if ( sDate is not None ):
        CheckAssert ( bBool = ( sDate in tDaysAvailable[ -1 ] ), sMsg = "Selected <sDate> is not available",
                      sExtraInfo = "Date: %s, Sensor: %s, (%s)" % ( sDate, sSensorID, str ( tDaysAvailable[ -1 ] ) ) )
        
    aSelectID = ( aID_All == tInfo[ 0 ] )
    aData_sel_ID = aDataAll[ aSelectID ]
    aDateTime_sel_ID = aDateTimeAll[ aSelectID ]
    
    aSelectTime = np.logical_and ( aDateTime_sel_ID >= CDateTimeStart, aDateTime_sel_ID < CDateTimeEnd )
    aDateTime = aDateTime_sel_ID[ aSelectTime ]
    aData = aData_sel_ID[ aSelectTime ]

    print ( ">> LoadData > Sensor %s; Filtered Data of Shape %s" % ( sSensorID, str ( aData.shape ) ) ) 
    print ( ">> LoadData > Sensor %s; Filtered Datetimes of Shape %s" % ( sSensorID, str ( aDateTime.shape ) ) )

    return ( aDateTime, aData, tInfo )
# ************************************ Grund-Routine: Einlesen der auf Festplatte abgelegten Daten *****************************************
def LoadRawData ( sDataType, sDate, sSubFolderMonthYear, bShowInfo = True ):
    sDataType = sDataType.capitalize ()
    if ( sDataType == "Matter" ):
        sSuffix = "_MATR"
    elif ( sDataType == "Temperature" ):
        sSuffix = "_TEMR"

    sInputFolder = GDictConfig.get ( "SaveFolder" )
    sBaseNameData = sInputFolder + sSubFolderMonthYear + "/" + "Day_Data" + sSuffix + ".npy"
    sBaseNameDateTime = sInputFolder + sSubFolderMonthYear + "/" + "Day_DateTime" + sSuffix + ".npy"
    sBaseNameID = sInputFolder + sSubFolderMonthYear + "/" + "Day_ID" + sSuffix + ".npy"
    sBaseNameDictInfo = sInputFolder + sSubFolderMonthYear + "/" + "Day_DictSensorInfo" + sSuffix + ".npy"
    sBaseNameDictDays = sInputFolder + sSubFolderMonthYear + "/" + "Day_DictSensorDays" + sSuffix + ".npy"
    
    CheckAssert ( bBool = ( ( os.path.isfile ( sBaseNameData ) == True ) and ( os.path.isfile ( sBaseNameDictInfo ) == True ) and 
                            ( os.path.isfile ( sBaseNameDateTime ) == True ) and ( os.path.isfile ( sBaseNameID ) == True ) ), 
                  sMsg = "Can't find Files!", 
                  sExtraInfo = "%s, %s, %s, %s" % ( sBaseNameData, sBaseNameDateTime, sBaseNameID, sBaseNameDictInfo ) )
    
    aDictInfo = np.load ( file = sBaseNameDictInfo, allow_pickle = True )
    DictInfo = dict ( aDictInfo.tolist () )
    aDictDays = np.load ( file = sBaseNameDictDays, allow_pickle = True )
    DictDays = dict ( aDictDays.tolist () )
    
    aDataAll = np.load ( file = sBaseNameData, allow_pickle = True )
    aDateTimeAll = np.load ( file = sBaseNameDateTime, allow_pickle = True )
    aID_All = np.load ( file = sBaseNameID, allow_pickle = True )
    aID = aID_All
    
    if ( sDate is not None ):
        sDateFormat = "%Y-%m-%dT%H:%M"
        sDateTimeStart = sDate + "T" + "00:00"
        CDateTimeStart = datetime.strptime ( sDateTimeStart, sDateFormat )
        CDateTimeEnd = CDateTimeStart +  timedelta ( days = 1 ) 
        aSelectTime = np.logical_and ( aDateTimeAll >= CDateTimeStart, aDateTimeAll <= CDateTimeEnd )
        
        aDateTimeAll = aDateTimeAll[ aSelectTime ]
        aDataAll = aDataAll[ aSelectTime ]
        aID = aID_All[ aSelectTime ]
        
    if ( aID.shape[ 0 ] != aID_All.shape[ 0 ] ):
        DictInfo_inv = { tTupel[ 0 ]: sKey for sKey, tTupel in DictInfo.items () }
        DictInfo_new = dict ()
        for iCountID in aID:
            sSensorID = DictInfo_inv.get ( iCountID )
            DictInfo_new[ sSensorID ] = DictInfo.get ( sSensorID )
        DictInfo = DictInfo_new
    
    if ( bShowInfo == True ):
        print ( ">> LoadRawData > Loaded Data of Shape %s" % ( str ( aDataAll.shape ) ) ) 
        print ( ">> LoadRawData > Loaded Datetimes of Shape %s" % ( str ( aDateTimeAll.shape ) ) )
        print ( ">> LoadRawData > Loaded ID Mapping of Shape %s" % ( str ( aID.shape  ) ) ) 
        print ( ">> LoadRawData > Loaded Dict with %d Keys" % ( len ( DictInfo.keys () ) ) ) 
    
    return ( aDateTimeAll, aDataAll, aID, DictInfo, DictDays )
# ************************** Ablegen der Daten einer Serie tDaySequence für ALLE IDs im npy Format auf Festplatte **************************
### einige Sensoren liefern 0.0 Werte (Sensor 8353) oder nur sehr wenige verschiedeen Werte (Sensor 12275); die können ausgeschlossen werden
def SaveData ( tDaySequence, sMonth, sYear, sDataType, sSubFolderMonthYear, bCheckAbnormality, bWriteFile, sLogFile = None ):
    sDataType = sDataType.capitalize ()
    
    if ( sDataType == "Matter" ):
        sSuffix = "_MATR"
    elif ( sDataType == "Temperature" ):
        sSuffix = "_TEMR"
        
    if ( sLogFile is not None ):
        CFile = open ( file = sLogFile, mode = "w+", encoding = "latin-1" )
    else:
        CFile = None

    DictHist = FindConsecutiveTimeSeries ( tDaySequence = tDaySequence, sMonth = sMonth, sYear = sYear, sDataType = sDataType )
    sInputFolder = GDictConfig.get ( "SaveFolder" )
    
    sBaseNameData = sInputFolder + sSubFolderMonthYear + "/" + "Day_Data" + sSuffix + ".npy"
    sBaseNameDateTime = sInputFolder + sSubFolderMonthYear + "/" + "Day_DateTime" + sSuffix + ".npy"
    sBaseNameID = sInputFolder + sSubFolderMonthYear + "/" + "Day_ID" + sSuffix + ".npy"
    sBaseNameDictInfo = sInputFolder + sSubFolderMonthYear + "/" + "Day_DictSensorInfo" + sSuffix + ".npy"
    sBaseNameDictDays = sInputFolder + sSubFolderMonthYear + "/" + "Day_DictSensorDays" + sSuffix + ".npy"
    
    iCounter = 1
    DictInfo = dict ()
    DictDays = dict ()
    ListData = list ()
    ListDateTime = list ()   
    ListID = list ()
    for sSensorID in DictHist.keys ():
        tDataAvailable = DictHist.get ( sSensorID ) ##  tTuple = ( alle Pfadangaben, alle Datumsangaben )
        bCheck, aDateTime, aData, tInfo, tDatesDataUsable = ImportDataSeriesSingleID ( iSensorCounter = iCounter, sSensorID = sSensorID, 
                                                                                       tDataAvailable = tDataAvailable, sDataType = sDataType, 
                                                                                       CFile = CFile, bCheckAbnormality = bCheckAbnormality )    
        if ( bCheck == False ):
            continue
        aID = np.full ( shape = ( aData.shape[ 0 ], ), dtype = np.int32, fill_value = iCounter )
        
        fLon, fLat, fLon_ref, fLat_ref = tInfo
        DictInfo[ sSensorID ] = ( ( iCounter, fLon, fLat, fLon_ref, fLat_ref ) )
        DictDays[ sSensorID ] = ( ( iCounter, tDatesDataUsable ) )
        ListData.append ( aData[ :, 1 : ] ) ## die IDs nicht noch einaml ablegen
        ListDateTime.append ( aDateTime )
        ListID.append ( aID )
        iCounter += 1
        
    if ( bWriteFile == True ):    
        aDataAll = np.vstack ( ListData )
        aDateTimeAll = np.hstack ( ListDateTime )
        aID_All = np.hstack ( ListID )
        
        if ( os.path.isfile ( sBaseNameData ) == False ):
            np.save ( file = sBaseNameData, arr = aDataAll )
            print ( ">> SaveData > Saved Data of Shape: %s" % ( str ( aDataAll.shape ) ) )
        else:
            print ( ">> SaveData > File %s already exists!" % ( sBaseNameData ) )
        
        if ( os.path.isfile ( sBaseNameDateTime ) == False ):
            np.save ( file = sBaseNameDateTime, arr = aDateTimeAll )
            print ( ">> SaveData > Saved Datetimes of Shape: %s" % ( str ( aDateTimeAll.shape ) ) )
        else:
            print ( ">> SaveData > File %s already exists!" % ( sBaseNameDateTime ) )    
                
        if ( os.path.isfile ( sBaseNameID ) == False ):
            np.save ( file = sBaseNameID, arr = aID_All )
            print ( ">> SaveData > Saved IDs of Shape: %s" % ( str ( aID_All.shape ) ) )
        else:
            print ( ">> SaveData > File %s already exists!" % ( sBaseNameID ) )   
                
        if ( os.path.isfile ( sBaseNameDictInfo ) == False ):
            np.save ( file = sBaseNameDictInfo, arr = DictInfo )
            print ( ">> SaveData > Saved DictInfo with %d Keys" % ( len ( DictInfo.keys () ) ) )
        else:
            print ( ">> SaveData > DictInfo %s already exists!" % ( sBaseNameDictInfo ) ) 
            
        if ( os.path.isfile ( sBaseNameDictDays ) == False ):
            np.save ( file = sBaseNameDictDays, arr = DictDays )
            print ( ">> SaveData > Saved DictDays with %d Keys" % ( len ( DictDays.keys () ) ) )
        else:
            print ( ">> SaveData > DictDays %s already exists!" % ( sBaseNameDictDays ) ) 
            
    if ( CFile is not None ):
        CFile.close ()             
        
    return
# ********************************* Auffinden von Datenserien für konsekutive Zeiten gegeben durch tDaySequence **************************** 
def FindConsecutiveTimeSeries ( tDaySequence, sMonth, sYear, sDataType, sBaseFolder = None ):
    sDataType = sDataType.capitalize ()
    CheckAssert ( bBool = ( sDataType in [ "Matter", "Temperature" ] ), sMsg = "Invalid Choice for <sDataType>!",
                  sExtraInfo = sDataType )
    
    sDay = "%d" % ( tDaySequence[ 0 ] )
    sDay = sDay.zfill ( 2 )
    
    if ( sBaseFolder is None ):
        sBaseFolder = GDictConfig.get ( "BaseFolderDay" ) 
    
    sStartFolder = sBaseFolder + sYear + sMonth + sDay + "/" + sDataType + "/"
    ListMatches = SearchFilesInFolder ( sSearchFolder = sStartFolder, sExtension = "csv", bIncludeSubDir = False, bReturnFullFilePath = False )

    DictHistory = dict ()
    for sFileName in ListMatches:
        tParts = sFileName.split ( "_" )
        tParts2 = tParts[ -1 ].split ( "." )
        sSensorID = tParts2[ 0 ]

        #iCounter = 0
        ListFilePath = list ()
        ListDate = list ()
        for iDay in tDaySequence:
            sDay = "%d" % ( iDay )
            sDay = sDay.zfill ( 2 )
            sNewFolder = sBaseFolder + sYear + sMonth + sDay + "/" + sDataType + "/"
            sDate = sYear + "-" + sMonth + "-" + sDay
            sNewFileName = sDate + "_" + tParts[ 1 ] + "_" + tParts[ 2 ] + "_" + tParts[ 3 ]
            sNewFilePath = sNewFolder + sNewFileName 
            bBool = os.path.isfile ( sNewFilePath ) 
            if ( bBool == True ):
                #iCounter += 1
                ListFilePath.append ( sNewFilePath )
                ListDate.append ( sDate )
            
        DictHistory[ sSensorID ] = ( tuple ( ListFilePath ), tuple ( ListDate) )
        
    return ( DictHistory )    
# ************************************** Aufbau der Daten-Serie über mehrere Tage für einen Sensor *****************************************
#                                       die Daten werden dabei von der Festplatte aus den CSV-Dateien gelesen
def ImportDataSeriesSingleID ( iSensorCounter, sSensorID, tDataAvailable, sDataType, CFile = None, bCheckAbnormality = False ):
    fLon_ref, fLat_ref = GDictConfig.get ( "CentralLocation" ) #11.575328, 48.137371 #München Marienplatz
    CCRS_geodetic = crs.Geodetic ()
    CCRS_azmequi = crs.AzimuthalEquidistant ( central_longitude = fLon_ref, central_latitude = fLat_ref )
    
    ListData = list ()
    ListDateTime = list ()
    ListDatesDataAvailable = list ()

    sDataType = sDataType.upper ()
    if ( sDataType in [ "MATTER", "M" ] ):
        uIndices = [ 3, 4 ]
    elif ( sDataType in [ "TEMPERATURE", "T" ] ):
        uIndices = 3

    for ik, sFilePath in enumerate ( tDataAvailable[ 0 ] ): ### tDataAvailable = ( alle Pfadangaben, alle Datumsangaben )
        if ( sDataType == "MATTER" ):
            bCheck, tInfo = ImportSensorDataMatter ( sFilePath = sFilePath, sSensorSpecification = "P1P2", CFile = CFile,
                                                     bCheckAbnormality = bCheckAbnormality ) 
        else:
            bCheck, tInfo = ImportSensorDataTemperature ( sFilePath = sFilePath )
        
        if ( bCheck == True ):
            sSensorID, sSensorType, ( fLat, fLon ), aRawData = tInfo
            aData = aRawData[ :, uIndices ]
            if ( aData.ndim == 1 ):
                aData = np.reshape ( aData, shape = ( aData.shape[ 0 ], 1 ) )

            aDates = np.asarray ( aRawData[ :, 2 ], dtype = np.datetime64 )
            aID = np.full ( shape = ( aData.shape[ 0 ], 1 ), dtype = np.float32, fill_value = float ( iSensorCounter ) )
            aData = np.hstack ( ( aID, aData ) )
            ListData.append ( aData )
            ListDateTime.append ( aDates )
            ListDatesDataAvailable.append ( ( tDataAvailable[ 1 ][ ik ] ) )
        else:
            PrintMessage ( sTextMsg = ">> ImportDataSeriesSingleID > Import failed for Sensor %s" % ( tInfo[ 0 ] ), sColor = "red", CFile = CFile )

    if ( ( len ( ListData ) == 0 ) or ( len ( ListDateTime ) == 0 ) ):
        PrintMessage ( sTextMsg = ">> ImportDataSeriesSingleID > No Files Found!" )
        
        return ( False, None, None, None, None )
    
    CheckAssert ( bBool = ( ( len ( ListData ) > 0 ) and ( len ( ListDateTime ) > 0 ) ), sMsg = "No Files Found!" )
    aDataAll = np.vstack ( ListData )
    aDateTimeAll = np.hstack ( ListDateTime )
    tCoord_azm = np.squeeze ( CCRS_azmequi.transform_points ( CCRS_geodetic, fLon, fLat ) )[ : 2 ] 
    
    print ( ">> ImportDataSeriesSingleID > Sensor ID: %s, Shape: %s" % ( sSensorID, str ( aDataAll.shape ) ) )

    return ( True, aDateTimeAll, aDataAll, ( fLon, fLat, tCoord_azm[ 0 ], tCoord_azm[ 1 ] ), tuple ( ListDatesDataAvailable ) )
# ************************************ Import und Konvertierung der Temperatur-Daten in Numpy lesbare Formate ******************************
def ImportSensorDataTemperature ( sFilePath ):
    sDateFormat = "%Y-%m-%dT%H:%M:%S"
    sFileName = os.path.splitext ( os.path.basename ( sFilePath ) )[ 0 ]
    tParts = sFileName.split ( "_" )
    sSensorType = tParts[ 1 ].upper ()
    sSensorID = tParts[ -1 ]
    CheckAssert ( bBool = ( sSensorType in [ "BME280", "BMP280", "BMP180", "DHT22", "SHT31" ] ), sMsg = "<sSensorType> is not configured!", 
                  sExtraInfo = sSensorType )

    tUseColumn, iHeaderLength, _ = GDictTemperatureColumns.get ( sSensorType, ( None, None ) )
    
    if ( ( tUseColumn is None ) and ( iHeaderLength is None ) ):
        return ( False, None )
    
    tIgnoreColumn = BuildComplement ( tSelectIndex = tUseColumn, iLength = iHeaderLength )
    _, ListCheck, ListData = AnalyzeDataStructure ( sDataFileName = sFilePath, sComments = None, sDelimiter = ";", tIgnoreColumn = tIgnoreColumn, 
                                                    bAddSummary = False, bAddInfoNA = False, bReturnCheckList = True )
    
    ## Prüfung auf nan Werte
    for ik in range ( len ( ListCheck ) ):
        tTupel = ListCheck[ ik ]
        if ( tTupel[ 2 ] != 0 ):
            print ( colored ( text = "Sensor %s of type %s includes %d NA Values!" % ( sSensorID, sSensorType, tTupel[ 2 ] ), color = "red", attrs = [ "bold" ] ) )
            return ( False, None )
    
    fLat = ListData[ 0 ][ 4 ]
    fLon = ListData[ 1 ][ 4 ]
        
    aDateStrings = ListData[ 2 ]
    aDates = np.zeros ( shape = aDateStrings.shape[ 0 ], dtype = "datetime64[s]" )
    for ik, sDate in enumerate ( aDateStrings ):
        try:
            CDateTime = datetime.strptime ( sDate, sDateFormat )
        except ValueError:
            print ( ik, CDateTime )
            
        aDates[ ik ] = CDateTime
    
    ListData[ 2 ] = aDates
    
    aRawData = np.transpose ( np.asarray ( ListData ) ) 
    
    return ( True, ( sSensorID, sSensorType, ( fLat, fLon ), aRawData ) )
# ********************************** Import und Konvertierung der Daten für Feinstaub in Numpy lesbare Formate *****************************
#                                   darüber hinnaus werden ggf. NA Werte entfernt !
## geändert: 10.06.2026: die Ersetzung von NA Werten in den Spalten für P1 und P2 wurde eingfügt
## geändert 14.06.2026: Daten mit zu wenigen verschiedenen Werten oder zeitlichen Unterbrechungen werden ignoriert
def ImportSensorDataMatter ( sFilePath, sSensorSpecification, tColumnCheckNA = ( 3, 4 ), CFile = None, bCheckAbnormality = False ):
    sSensorSpecification = sSensorSpecification.upper ()
    sDateFormat = "%Y-%m-%dT%H:%M:%S"
    fMaxInterruptionMinAllowed = GDictConfig.get ( "MaxAllowedInterruption" )
    iMinNumDifferentValues = GDictConfig.get ( "MinimalNumDifferentValues" )
    
    sFileName = os.path.splitext ( os.path.basename ( sFilePath ) )[ 0 ]
    tParts = sFileName.split ( "_" )
    sSensorType = tParts[ 1 ].upper ()
    sSensorID = tParts[ -1 ]
    
    CheckAssert ( bBool = ( sSensorType in [ "SDS011" ] ), sMsg = "<sSensorType> is not configured!", sExtraInfo = sSensorType )
    CheckAssert ( bBool = ( sSensorSpecification in [ "P1", "P2", "P1P2" ] ), sMsg = "Invalid Choice for <sSensorSpecification>!" )
    
    sKey = "Indices" + sSensorSpecification
    tUseColumn, iHeaderLength, _ = GDictMatterColumns.get ( sKey, ( None, None ) )
    
    if ( ( tUseColumn is None ) and ( iHeaderLength is None ) ):
        return ( False, None )
    
    tIgnoreColumn = BuildComplement ( tSelectIndex = tUseColumn, iLength = iHeaderLength )
    _, ListCheck, ListData = AnalyzeDataStructure ( sDataFileName = sFilePath, sComments = None, sDelimiter = ";", tIgnoreColumn = tIgnoreColumn, 
                                                    bAddSummary = False, bAddInfoNA = False, bReturnCheckList = True )
    
    fLat = ListData[ 0 ][ 4 ]
    fLon = ListData[ 1 ][ 4 ]
        
    aDateStrings = ListData[ 2 ]
    aDates = np.zeros ( shape = aDateStrings.shape[ 0 ], dtype = "datetime64[s]" )
    for ik, sDate in enumerate ( aDateStrings ):
        try:
            CDateTime = datetime.strptime ( sDate, sDateFormat )
        except ValueError:
            print ( ik, CDateTime )
            
        aDates[ ik ] = CDateTime
    
    ListData[ 2 ] = aDates
    aRawData = np.transpose ( np.asarray ( ListData ) ) 
    
    if ( tColumnCheckNA is not None ):
        CheckAssert ( bBool = ( len ( tColumnCheckNA ) == 2 ), sMsg = "Invalid Shape of <tColumnCheckNA> !" )
        aIndicesSelect2Columns = ( ~np.isnan ( np.asarray ( aRawData[ :, tColumnCheckNA ], dtype = np.float32  ) ) )
        aIndicesSelect = np.logical_and ( aIndicesSelect2Columns[ :, 0 ], aIndicesSelect2Columns[ :, 1 ] )

        if ( np.sum ( aIndicesSelect ) != aRawData.shape[ 0 ] ):
            PrintMessage ( sTextMsg = ">> ImportSensorDataMatter > Data File for ID %s contains NA-Values" % ( sSensorID ),
                           sColor = "red", CFile = CFile )
            aDates = aDates[ aIndicesSelect ]
            aRawData = aRawData[ aIndicesSelect ]

            ### Check, ob es funktioniert hat
            aIndicesSelect2Columns = ( ~np.isnan ( np.asarray ( aRawData[ :, tColumnCheckNA ], dtype = np.float32  ) ) )
            aIndicesSelect = np.logical_and ( aIndicesSelect2Columns[ :, 0 ], aIndicesSelect2Columns[ :, 1 ] )

            CheckAssert ( bBool = ( np.sum ( aIndicesSelect ) == aRawData.shape[ 0 ] ), sMsg = "Imputation failed!" )
            PrintMessage ( sTextMsg = ">> ImportSensorDataMatter > NA-Values in Data File ID %s removed" % ( sSensorID ), 
                           sColor = "green", CFile = CFile )
               
    ## Überprüfung auf zeitliche Unterbrechungen UND zu wenige verschiedene Werte
    if ( bCheckAbnormality == True ):
        aData_check = np.asarray ( aRawData[ :, 3 ], dtype = np.float64 )
        DictResult = ScreenDataSeriesAbnormality ( aData = aData_check, aDateTime = aDates )
        iNumDifferentValues = DictResult.get ( "NumberDifferentValues" )
        fMaxInterruptionMin = ( DictResult.get ( "TimeDeltaQuartiles" )[ 4 ] / 60.0 )
        
        if ( ( iNumDifferentValues < iMinNumDifferentValues ) or ( fMaxInterruptionMin > fMaxInterruptionMinAllowed ) ): 
            sText = "Sensor %s failed check on %s: [Breaks:%.0f, #Values:%d]!" % ( sSensorID, str ( aDates[ 0 ] )[ : 10 ], fMaxInterruptionMin, iNumDifferentValues )
            PrintMessage ( sTextMsg = ">> ImportSensorDataMatter > " + sText, sColor = "red", CFile = CFile )
            
            return ( False, ( sSensorID, ) )
                
    return ( True, ( sSensorID, sSensorType, ( fLat, fLon ), aRawData ) )
# ************************ Berechung von Mittelwert und Median für JEDEN Tag aller vorhandener Daten nach Bereinigung **********************
def SaveDataStatisticDay ( sDataType, sSubFolderMonthYear, bWriteFile ):
    sDataType = sDataType.capitalize ()
    fThreshold, iMinimalLengthSequence, iSumConsecutiveMeasurementsAbove = GDictConfig.get ( "ThresholdExcessParameter" )
    
    if ( sDataType == "Matter" ):
        sSuffix = "_MATR"
    elif ( sDataType == "Temperature" ):
        sSuffix = "_TEMR"

    sInputFolder = GDictConfig.get ( "SaveFolder" )
    sBaseNameStatisticDayDateTime = sInputFolder + sSubFolderMonthYear + "/" + "AvgDay_DateTime" + sSuffix + ".npy"
    sBaseNameStatisticDayData = sInputFolder + sSubFolderMonthYear + "/" + "AvgDay_Data" + sSuffix + ".npy"
   
    aDateTimeAll, aDataAll, aID, DictInfo, DictDays = LoadRawData ( sDataType = sDataType, sDate = None, sSubFolderMonthYear = sSubFolderMonthYear )

    ListDates = list ()
    ListAverage = list ()
    sDateFormat = "%Y-%m-%d"
    
    for sKey in DictDays.keys ():
        tInfo = DictDays.get ( sKey )
        iID = tInfo[ 0 ]
        tAvailableDays = tInfo[ 1 ]
    
        aSelectID = ( aID == iID )
        aData_ID = np.asarray ( aDataAll[ aSelectID ], dtype = np.float32 )
        aDateTime_ID = aDateTimeAll[ aSelectID ]

        for sDate in tAvailableDays:
            aMean, aMedian = GetDataDay ( sDate = sDate, aDateTime = aDateTime_ID, aData = aData_ID, bReturnData = False )
            CDate = datetime.strptime ( sDate, sDateFormat )
            
            DictResult = ScreenDataSeriesAbnormality ( aData = aData_ID[ :, 1 ], fThreshold = fThreshold ) # PM2.5
            aThresholdSequenceLength = DictResult[ "SequenceLengthThreshold" ] 
            if ( aThresholdSequenceLength is not None ):
                aThresholdSequenceLength = aThresholdSequenceLength[ aThresholdSequenceLength > iMinimalLengthSequence ]
                if ( np.sum ( aThresholdSequenceLength ) > iSumConsecutiveMeasurementsAbove ):
                    print ( ">> ID: %s, date: %s, #cons. measurements above %.0f: %d [%d], mean: %s, median: %s" 
                            % ( sKey, sDate, fThreshold, np.sum ( aThresholdSequenceLength ), iSumConsecutiveMeasurementsAbove, 
                                str ( aMean ), str ( aMedian ) ) )
                    continue
                
            ListDates.append ( CDate )
            ListAverage.append ( ( iID, aMean[ 0 ], aMean[ 1 ], aMedian[ 0 ], aMedian[ 1 ] ) )

    if ( bWriteFile == True ):    
        ## iID, MW PM10, MW PM2.5, Median PM10, Median PM2.5
        aAverageDataAll = np.asarray ( ListAverage ) 
        aAverageDateTimeAll = np.hstack ( ListDates )
        
        if ( os.path.isfile ( sBaseNameStatisticDayData ) == False ):
            np.save ( file = sBaseNameStatisticDayData, arr = aAverageDataAll )
            print ( ">> SaveDataStatisticDay > Saved Data of Shape: %s" % ( str ( aAverageDataAll.shape ) ) )
        else:
            print ( ">> SaveDataStatisticDay > File %s already exists!" % ( sBaseNameStatisticDayData ) )    
        
        if ( os.path.isfile ( sBaseNameStatisticDayDateTime ) == False ):
            np.save ( file = sBaseNameStatisticDayDateTime, arr = aAverageDateTimeAll )
            print ( ">> SaveDataStatisticDay > Saved Datetimes of Shape: %s" % ( str ( aAverageDateTimeAll.shape ) ) )
        else:
            print ( ">> SaveDataStatisticDay > File %s already exists!" % ( sBaseNameStatisticDayDateTime ) )
        
    return

def LoadDataStatisticDay ( sDataType, sSubFolderMonthYear ):
    sDataType = sDataType.capitalize ()
    if ( sDataType == "Matter" ):
        sSuffix = "_MATR"
    elif ( sDataType == "Temperature" ):
        sSuffix = "_TEMR"

    sInputFolder = GDictConfig.get ( "SaveFolder" )
    sBaseNameStatisticDayDateTime = sInputFolder + sSubFolderMonthYear + "/" + "AvgDay" + "_DateTime" + sSuffix + ".npy"
    sBaseNameStatisticDayData = sInputFolder + sSubFolderMonthYear + "/" + "AvgDay" + "_Data" + sSuffix + ".npy"
    sBaseNameDictInfo = sInputFolder + sSubFolderMonthYear + "/" + "Day_DictSensorInfo" + sSuffix + ".npy"
   
    
    CheckAssert ( bBool = ( ( os.path.isfile ( sBaseNameStatisticDayDateTime ) == True ) and 
                            ( os.path.isfile ( sBaseNameStatisticDayData ) == True ) and 
                            ( os.path.isfile ( sBaseNameDictInfo ) == True ) ), 
                  sMsg = "Can't find Files!", 
                  sExtraInfo = "%s, %s, %s" % ( sBaseNameStatisticDayDateTime, sBaseNameStatisticDayData, sBaseNameDictInfo ) )
    
    aDictInfo = np.load ( file = sBaseNameDictInfo, allow_pickle = True )
    DictInfo = dict ( aDictInfo.tolist () )

    aAverageDataAll = np.load ( file = sBaseNameStatisticDayData, allow_pickle = True )
    aAverageDateTimeAll = np.load ( file = sBaseNameStatisticDayDateTime, allow_pickle = True )
    aID = aAverageDataAll[ :, 0 ]
    ## MW PM10, MW PM2.5, Median PM10, Median PM2.5
    aAverageDataAll = aAverageDataAll[ :, 1 : ]
    
    return ( aAverageDateTimeAll, aAverageDataAll, aID, DictInfo )

def CalcAverages ( aDateTime, aData, sStartDate, sStartTime, iWindowDelta, sDeltaUnit ):
    CheckAssert ( bBool = ( ( len ( sStartTime ) == 5 ) and ( sStartTime[ 2 ] == ":" ) ), sMsg = "Invalid Format <sStartTime>!" )
    CheckAssert ( bBool = ( aDateTime.shape[ 0 ] == aData.shape[ 0 ] ), sMsg = "Shape Mismatch!" )
    CheckAssert ( bBool = ( isinstance ( aDateTime[ - 1 ], np.datetime64 ) ), sMsg = "Invalid Format <aDateTime>!",
                  sExtraInfo = "%s" % ( type ( aDateTime[ - 1 ] ) ) )
    
    sDeltaUnit = sDeltaUnit.lower ()
    CheckAssert ( bBool = ( sDeltaUnit in [ "days", "hours" ] ), sMsg = "Invalid Choice for <sDeltaUnit>!" )
    sDateFormat = "%Y-%m-%dT%H:%M"
    
    if ( sDeltaUnit == "days" ):
        CTimeDelta = timedelta ( days = iWindowDelta ) 
    elif ( sDeltaUnit == "hours" ):
        CTimeDelta = timedelta ( hours = iWindowDelta ) 
    
    sDateTimeStart = sStartDate + "T" + sStartTime
    CDateTimeStart = datetime.strptime ( sDateTimeStart, sDateFormat )
    CDateTimeEnd = CDateTimeStart +  CTimeDelta 
    
    fMaxTime = np.amax ( aDateTime )
    #print ( fMaxTime )

    ListAverages = list ()
    ListDates = list ()
    while ( CDateTimeStart < fMaxTime ):
        aSelectTime = np.logical_and ( aDateTime >= CDateTimeStart, aDateTime <= CDateTimeEnd )
        aData_sel = aData[ aSelectTime ]
        if ( aData_sel.shape[ 0 ] > 0 ):
            aMean, aMedian = np.mean ( aData_sel, axis = 0 ), np.median ( aData_sel, axis = 0 )
            ListDates.append ( CDateTimeStart )
            ListAverages.append ( ( aMean[ 0 ], aMean[ 1 ], aMedian[ 0 ], aMedian[ 1 ] ) )
        else:    
            sText = "Start: %s, End: %s, Shape: %s" % ( str ( CDateTimeStart ), str ( CDateTimeEnd ), str ( aData_sel.shape ) ) 
            print ( ">> CalcAverages > No data found; " + sText ) 
                    
        CDateTimeStart = CDateTimeEnd
        CDateTimeEnd = CDateTimeEnd + CTimeDelta
        
    aDateTime = np.asarray ( ListDates, dtype = np.datetime64 )
    aAverageData = np.asarray ( ListAverages )
    
    return ( aDateTime, aAverageData )
# ****************************** Berechung von Mittelwert und Median aller vorhandener Daten nach Bereinigung ******************************
def SaveDataStatisticMonth ( sDataType, sSubFolderMonthYear = "Apr2026" ):
    sDataType = sDataType.capitalize ()
    fThreshold, iMinimalLengthSequence, iSumConsecutiveMeasurementsAbove = GDictConfig.get ( "ThresholdExcessParameter" )
    
    if ( sDataType == "Matter" ):
        sSuffix = "_MATR"
    elif ( sDataType == "Temperature" ):
        sSuffix = "_TEMR"

    sInputFolder = GDictConfig.get ( "SaveFolder" )
    sBaseNameDictStatistic = sInputFolder + sSubFolderMonthYear + "/" + "AvgMonth_Dict" + sSuffix + ".npy"

    aDateTimeAll, aDataAll, aID, DictInfo, _ = LoadRawData ( sDataType = sDataType, sDate = None, sSubFolderMonthYear = sSubFolderMonthYear, bShowInfo = False )

    DictStatistic = dict ()
    for sKey in DictInfo.keys ():
        tInfo = DictInfo.get ( sKey )
        iID = tInfo[ 0 ]
        aSelectID = ( aID == iID )
        aData_ID = np.asarray ( aDataAll[ aSelectID ], dtype = np.float32 )
        aMedian = np.median ( aData_ID, axis = 0 )
        aMean = np.mean ( aData_ID, axis = 0 )
        
        DictResult = ScreenDataSeriesAbnormality ( aData = aData_ID[ :, 1 ], fThreshold = fThreshold ) # PM2.5
        aThresholdSequenceLength = DictResult[ "SequenceLengthThreshold" ] 
        if ( aThresholdSequenceLength is not None ):
            aThresholdSequenceLength = aThresholdSequenceLength[ aThresholdSequenceLength > iMinimalLengthSequence ]
            if ( np.sum ( aThresholdSequenceLength ) > iSumConsecutiveMeasurementsAbove ):
                print ( ">> ID: %s, #cons. measurements above %.0f: %d [%d], mean: %s, median: %s" 
                        % ( sKey, fThreshold, np.sum ( aThresholdSequenceLength ), iSumConsecutiveMeasurementsAbove, str ( aMean ), str ( aMedian ) ) )
                continue
        
        DictStatistic[ sKey ] = ( tInfo, ( aMean, aMedian ) )
        
    if ( os.path.isfile ( sBaseNameDictStatistic ) == False ):
        np.save ( file = sBaseNameDictStatistic, arr = DictStatistic )
        print ( ">> SaveDataStatistic > Saved Dict with %d Keys" % ( len ( DictStatistic.keys () ) ) )
    else:
        print ( ">> SaveDataStatistic > Dict %s already exists!" % ( sBaseNameDictStatistic ) ) 
    
    return

def LoadDataStatisticMonth ( sDataType, sSubFolderMonthYear, sAverageParameter = "mean" ):
    sDataType = sDataType.capitalize ()
    sAverageParameter = sAverageParameter.lower ()
    CheckAssert ( bBool = ( sAverageParameter in [ "mean", "median" ] ), sMsg = "Invalid Choice for <sAverageParameter>!" )
    
    if ( sDataType == "Matter" ):
        sSuffix = "_MATR"
        iArrayLength = 6
    elif ( sDataType == "Temperature" ):
        sSuffix = "_TEMR"
        iArrayLength = 5

    sInputFolder = GDictConfig.get ( "SaveFolder" )
    sBaseNameDictStatistic = sInputFolder + sSubFolderMonthYear + "/" + "AvgMonth_Dict" + sSuffix + ".npy"
        
    aDictStatistic = np.load ( file = sBaseNameDictStatistic, allow_pickle = True )
    DictStatistic = dict ( aDictStatistic.tolist () )
    
    aRawData = np.zeros ( shape = ( len ( DictStatistic ), iArrayLength ), dtype = np.float64 )
    for ik, sKey in enumerate ( DictStatistic.keys () ):
        tInfo, tStatistic = DictStatistic.get ( sKey )
        ### Aufpassen ! Die Reihenfolge von Lat und Lon ist andersherum als in der DictInfo hinterlegt
        if ( sAverageParameter == "mean" ):
            aRawData[ ik ] = tInfo[ 2 ], tInfo[ 1 ], tInfo[ 4 ], tInfo[ 3 ], tStatistic[ 0 ][ 0 ], tStatistic[ 0 ][ 1 ]
        elif ( sAverageParameter == "median" ):
            aRawData[ ik ] = tInfo[ 2 ], tInfo[ 1 ], tInfo[ 4 ], tInfo[ 3 ], tStatistic[ 1 ][ 0 ], tStatistic[ 1 ][ 1 ]

    return ( aRawData, DictStatistic )
# **************************************** Ablegen aller Locations der SensorCommunity und LUA Sensoren ************************************
def SaveAllSensorLocations ( sSubFolderMonthYear ):
    if ( sSubFolderMonthYear == "Apr2026" ):
        sBaseNameDictAllSensorLocations = "C:/DATA/Daten/Kriging/SensorCommunity/Apr2026/Dict_LOC.npy"
    elif ( sSubFolderMonthYear == "Nov2025" ):
        sBaseNameDictAllSensorLocations = "C:/DATA/Daten/Kriging/SensorCommunity/Nov2025/Dict_LOC.npy"
    elif ( sSubFolderMonthYear == "Jul2025" ):
        sBaseNameDictAllSensorLocations = "C:/DATA/Daten/Kriging/SensorCommunity/Jul2025/Dict_LOC.npy"
        
    DictSensorLocation = dict ()

    _, DictInfoSC = LoadDataStatisticMonth ( sDataType = "matter", sSubFolderMonthYear = sSubFolderMonthYear )
    for sKey in DictInfoSC.keys ():
        tInfo, tStat = DictInfoSC.get ( sKey )
        iCounterId, fLat, fLon, fLat_ref, fLon_ref = tInfo
        DictSensorLocation[ sKey ] = ( iCounterId, fLat, fLon, fLat_ref, fLon_ref )
    
    _, _, DictInfoLUA = ImportSensorDataMatterLUA ( sMonthYear = "Apr2026" )
    for sKey in DictInfoLUA.keys ():
        tInfo = DictInfoLUA.get ( sKey )
        iCounterId, fLat, fLon, fLat_ref, fLon_ref = tInfo
        DictSensorLocation[ sKey ] = ( iCounterId, fLat, fLon, fLat_ref, fLon_ref )
       
    np.save ( file = sBaseNameDictAllSensorLocations, arr = DictSensorLocation )

    return

def LoadAllSensorLocations ():
    sBaseNameDictAllSensorLocations = "C:/DATA/Daten/Kriging/SensorCommunity/Apr2026/Dict_LOC.npy"
    aDictSensorLocation = np.load ( file = sBaseNameDictAllSensorLocations, allow_pickle = True )
    DictSensorLocation = dict ( aDictSensorLocation.tolist () )
    
    return ( DictSensorLocation )
    
# ****************************************************** Auffinden aller gültigen Dateien **************************************************
def ListAllSensorFiles ( sSearchFolder, bReturnFullFilePath = True ):
    CheckAssert ( bBool = ( os.path.isdir ( sSearchFolder ) == True ), sMsg = "<sSearchFolder> must be a Directory!" ) 

    ListAllItems = glob.glob ( pathname = "*.csv", root_dir = sSearchFolder, dir_fd = None, recursive = False, include_hidden = True )   
    
    if ( bReturnFullFilePath == True ):
        ListAllItems = [ os.path.join ( sSearchFolder, sFileName ) for sFileName in ListAllItems ]
        
    ListAllItems = sorted ( ListAllItems, key = str.lower )
    
    return ( ListAllItems )
# *********************** Kopieren der nutzbaren Dateien in einen Unterordner (z.B. "DataMatter" oder "DataTemp" ) *************************    
def SelectUsableFiles ( sFolder, sCopyFolder, sDataType, CMultiPolygonArea = None ):
    if ( CMultiPolygonArea is None ):
        _, CMultiPolygonArea = ShowBorder ( sPlaceName = "Munich, Germany", sGraphType = "none" )
        
    iCounter = 0
    if ( sCopyFolder is not None ):
        CheckAssert ( bBool = ( os.path.isdir ( sCopyFolder ) ), sMsg = "<sCopyFolder> must be an existing Folder" )
    else:
        sDataType = sDataType.lower ()
        sCopyFolder = sFolder + "/" + sDataType.capitalize ()
        
    ListAllFiles = SearchFilesInFolder ( sSearchFolder = sFolder, sExtension = "csv", sFilePattern = None, 
                                         bIncludeSubDir = False, bReturnFullFilePath = True )
    for sFilePath in ListAllFiles:
        bReturnValue = CheckUsableData ( sFilePath = sFilePath, CMultiPolygonArea = CMultiPolygonArea, sDataType = sDataType,
                                         iNumMinimumLines = 40 )
        if ( bReturnValue ):
            if ( sCopyFolder is not None ):
                shutil.copy2 ( sFilePath, sCopyFolder )
            iCounter += 1
            
    print ( "Found %d valid Files" % ( iCounter ) )
    
    return
# ***************************** Download der Dateien aus einem URL-Ordner, die in einer Region sPlaceName liegen ***************************
def DownloadFiles ( sPlaceName = "Munich, Germany", sSaveFolder = None, sFileContainingAllLinks = None, sDataType = None, iStartLink = None ):
    if ( ( sSaveFolder is None ) or ( len ( sSaveFolder ) == 0 ) ):
        sSaveFolder = GDictConfig.get ( "DefaultSaveFolder" )
        
    iCounter = 0
    _, CMulPolyArea = ShowBorder ( sPlaceName = sPlaceName, sGraphType = "none" )
    
    with open ( sFileContainingAllLinks, "r" ) as CFileLinks:
        ListAllURL = CFileLinks.readlines ()
        if ( iStartLink is not None ):
            ListAllURL = ListAllURL[ iStartLink : ]
        for sFileURL in tqdm.tqdm ( ListAllURL ):
            sFileURL = sFileURL.strip ()
            tParts = sFileURL.split ( "/" )
            sFileBaseName = tParts[ -1 ]
            sFileName = sSaveFolder + "/" + sFileBaseName 
            
            if ( os.path.isfile ( sFileName ) == True ):
                continue
                
            CResponse = requests.get ( sFileURL, stream = True )   
            sResponseText = CResponse.text
                
            if ( CResponse.ok == False ):
                print ( ">> DownloadFiles > Response for File %s was not ok! (%d)" % ( sFileBaseName, CResponse.status_code ) )
                continue

            ListResponseTextLines = sResponseText.splitlines ()
            sHeader, sFirstRow = ListResponseTextLines[ 0 ], ListResponseTextLines[ 1 ]
            iCheck1, bCheck2 = CheckValidData ( sLine1 = sHeader, sLine2 = sFirstRow, CMultiPolygonArea = CMulPolyArea, 
                                                sDataType = sDataType )
                
            if ( bCheck2 == True ):
                with open ( sFileName, "wb" ) as CFileData:
                    for aRawData in CResponse.iter_content ( chunk_size = 1024 ):
                        CFileData.write ( aRawData )
                        iCounter += 1
                            
                print ( "\nDownloaded file %s (Chk Px: %d, Chk Munich: %s)" % ( sFileBaseName, iCheck1, bCheck2 ) )
                CFileData.close ()
                
        CFileLinks.close ()
        
    print ( ">> DownloadFiles > Analyzed %d Files" % ( iCounter ) )

    return  
# ************************** Prüfung: Daten enthalten Px Messungen OUTDOOR UND liegen im Multi-Polygon der Area UND ************************
###                            haben eine bestimmte Mindestlänge 
def CheckUsableData ( sFilePath, CMultiPolygonArea, sDataType, iNumMinimumLines = 40 ):
    sDataType = sDataType.upper ()
    CheckAssert ( bBool = ( sDataType in [ "MATTER", "M", "TEMPERATURE", "T" ] ), sMsg = "Invalid Choice for <sdataType>!" )
    CheckAssert ( bBool = ( isinstance ( CMultiPolygonArea, shapely.geometry.multipolygon.MultiPolygon ) ), 
                  sMsg = "Wrong Format for <CMultiPolygonArea>!" )
    
    tInvalidSubStrings = ( "indoor", "laerm", "radiation" )
    bReturnValue = False
    
    if ( sDataType in [ "MATTER", "M" ] ):
        iMinValueCheck1 = 2
    elif ( sDataType in [ "TEMPERATURE", "T" ] ):
        iMinValueCheck1 = 1
    
    sFileName = os.path.splitext ( os.path.basename ( sFilePath ) )[ 0 ]
    iCheck = np.sum ( np.asarray ( tuple ( map ( lambda sSub : sSub in sFileName, tInvalidSubStrings ) ) ) )
    if ( iCheck > 0 ):
        return ( False )

    with open ( sFilePath, "r" ) as CFile:
        aRawData = CFile.readlines ()
        sHeader, sFirstRow = aRawData[ 0 ], aRawData[ 1 ]
        iCheckType, bCheckArea = CheckValidData ( sLine1 = sHeader, sLine2 = sFirstRow, CMultiPolygonArea = CMultiPolygonArea, 
                                                  sDataType = sDataType )
        
        if ( ( iCheckType >= iMinValueCheck1 ) and ( bCheckArea == True ) and ( len ( aRawData ) >= iNumMinimumLines ) ):
            bReturnValue = True
        
        CFile.close ()
        
    return ( bReturnValue )
# ************************** Prüfung ob Daten die gewünschten Messungen enthalten UND im Multi-Polygon der Area liegen *********************
def CheckValidData ( sLine1, sLine2, CMultiPolygonArea, sDataType ):
    sDataType = sDataType.upper ()
    CheckAssert ( bBool = ( sDataType in [ "MATTER", "TEMPERATURE", "ANY" ] ), sMsg = "Invalid Choice for <sdataType>!" )
    
    CheckAssert ( bBool = ( isinstance ( CMultiPolygonArea, shapely.geometry.multipolygon.MultiPolygon ) ), 
                  sMsg = "Wrong Format for <CMultiPolygonArea>!" )
    
    if ( sDataType == "MATTER" ):
        iCheckType = np.sum ( np.asarray ( tuple ( map ( lambda sKeyWord: sKeyWord in sLine1, 
                                                      GDictConfig.get ( "HeaderKeywordsMatter" ) ) ) ) )
    elif ( sDataType == "TEMPERATURE" ): 
        iCheckType = int ( sLine1.find ( GDictConfig.get ( "HeaderKeywordTemperature" ) ) != -1 ) 
    else:
        iCheckType = -1 # None ist nicht gut, da der Wert später mit Format %d ausgegeben wird

    tParts = sLine2.split ( ";" )
    fLat, fLon = float ( tParts[ 3 ] ), float ( tParts[ 4 ] )

    CPoint = shapely.geometry.Point ( fLon, fLat ) # das ist die richtige Reihenfolge!
    bCheckArea = CMultiPolygonArea.intersects ( CPoint )
        
    return ( iCheckType, bCheckArea )
# ******************************************** Auffinden aller Dateien in einem bestimmten URL-Ordner **************************************
def ListAllRefURL ( sURL, sExt = "csv", sSaveFilePath = None ):
    CResponse = requests.get ( sURL )
    if ( CResponse.ok == True ):
        sResponseText = CResponse.text
    else:
        return ( CResponse.raise_for_status () )
    
    CSoup = BeautifulSoup ( sResponseText, "html.parser" )
    ListRef = [ sURL + node.get ( "href" ) for node in CSoup.find_all ( "a" ) if node.get ( "href" ).endswith ( sExt ) ]
    
    if ( sSaveFilePath is not None ):
        with open ( sSaveFilePath, "w+" ) as CFile:
            for sLine in ListRef:
                 CFile.write ( sLine + "\n" )
        CFile.close ()
     
    return ( ListRef )
# ************************************************************* Hilfs-Funktion *************************************************************
def BuildComplement ( tSelectIndex, iLength ):
    aResult = np.arange ( start = 0, stop = iLength )
    aMask = np.ones ( shape = iLength, dtype = bool )
    aMask[ np.asarray ( tSelectIndex ) ] = False
    
    return ( aResult[ aMask ] )
# ************************** Importieren der aufgearbeiteten Sensordaten des Landesumweltamtes Bayern in München ***************************
def ImportSensorDataMatterLUA ( sMonthYear ):
    fLon_ref, fLat_ref = GDictConfig.get ( "CentralLocation" ) #11.575328, 48.137371 #München Marienplatz
    CCRS_geodetic = crs.Geodetic ()
    CCRS_azmequi = crs.AzimuthalEquidistant ( central_longitude = fLon_ref, central_latitude = fLat_ref )
    
    sBaseFolder = GDictConfig.get ( "BaseFolderMonthLUA" )# "C:/DATA/Daten/Kriging/Landesumweltamt Bayern/"
    sFilePath = sBaseFolder + "PM2x5_und_PM10_München_" + sMonthYear + ".txt"
    sDateFormat = "%d.%m.%Y %H:%M"
    DictInfo = dict ()
    
    DictBasicInfo = { "Joh"  : ( "Johanneskirchen", ( 48.17319, 11.64804 ) ), # Lat, Lon
                      "Lan"  : ( "Landshuter Allee", (  48.14955, 11.53653 ) ),
                      "Lot"  : ( "Lothstraße", (  48.15455, 11.55466 ) ),
                      "Sta"  : ( "Stacchus", (  48.13732, 11.56481 ) ) }
    
    aDescriptions, ListData = AnalyzeDataStructure ( sDataFileName = sFilePath, tIgnoreColumn = None, tGermanFormatColumn = None, 
                                                     sComments = "#", sDelimiter = "\t", bAddSummary = False, bAddInfoNA = False, 
                                                     bReturnCheckList = False )
    print ( aDescriptions )
    
    for ik, sKey in enumerate ( DictBasicInfo.keys () ):
        tInfo = DictBasicInfo.get ( sKey )
        fLat, fLon = np.asarray ( tInfo[ 1 ], dtype = np.float64 )
        fLon_ref, fLat_ref = np.squeeze ( CCRS_azmequi.transform_points ( CCRS_geodetic, fLon, fLat ) )[ : 2 ] 
        DictInfo[ sKey ] = ( ik + 1001, fLon, fLat, fLon_ref, fLat_ref )
        
    aDateStrings = ListData[ 0 ]
    aDates = np.zeros ( shape = aDateStrings.shape[ 0 ], dtype = "datetime64[s]" )
    for ik, sDate in enumerate ( aDateStrings ):
        try:
            CDateTime = datetime.strptime ( sDate, sDateFormat )
        except ValueError:
            print ( ik, CDateTime )
        
        aDates[ ik ] = CDateTime
    
    ListData[ 0 ] = aDates
    aRawData = np.transpose ( np.asarray ( ListData ) ) 
    ## Joh 2.5, Joh 10, Lan 2.5, Lan 10, Lot 2.5, Lot 10, Sta 2.5, Sta 10
    aData = np.asarray ( aRawData[ :, 1 : ], dtype = np.float32 )
    aDateTime = np.asarray ( aRawData[ :, 0 ], dtype = np.datetime64 )

    return ( aDateTime, aData, DictInfo )
# ********************** Selektion eines zeitlichen Intervalls innerhalb der Daten und Berechnung von Mean und Median **********************
def GetDataDay ( sDate, aDateTime, aData, bReturnData = True ):
    sDateFormat = "%Y-%m-%dT%H:%M"
    sDateTimeStart = sDate + "T" + "00:00"
    CDateTimeStart = datetime.strptime ( sDateTimeStart, sDateFormat )
    CDateTimeEnd = CDateTimeStart +  timedelta ( days = 1 ) 
    
    aSelectTime = np.logical_and ( aDateTime >= CDateTimeStart, aDateTime <= CDateTimeEnd )
    aDateTime_sel = aDateTime[ aSelectTime ]
    aData_sel = aData[ aSelectTime ]
    
    if ( bReturnData == True ):
        return ( aDateTime_sel, aData_sel, np.mean ( aData_sel, axis = 0 ), np.median ( aData_sel, axis = 0 ) )
    else:
        return ( np.mean ( aData_sel, axis = 0 ), np.median ( aData_sel, axis = 0 ) )
# ******************************** Bestimmung der Pfade zu den Speicherorten, an denen die Daten abgekegt sind *****************************
def GetDataFolderPath ( sDate, sDataType = "matter" ):
    sDataType = sDataType.capitalize ()
    iMonth = int ( sDate[ 5 : 7 ] )
    sMonth = GetMonth ( iChoice = iMonth )[ : 3 ]
    sYear = sDate[ : 4 ]
    sBaseFolder = "C:/DATA/Daten/Kriging/SensorCommunity/" + sMonth + sYear + "/"
    sFolderPath = sBaseFolder + sDataType

    return ( sFolderPath )
# **************************************** Ausgabe der Meldungen auf Bildschirm oder in Log-Datei ******************************************
def PrintMessage ( sTextMsg, sColor = "none", CFile = None ):
    sColor = sColor.lower ()
    CheckAssert ( bBool = ( sColor in [ "none", "red", "green" ] ), sMsg = "Invalid Parameter <sColor>!" )
    
    if ( CFile is not None ):
        print ( sTextMsg, file = CFile )
    else:
        if ( sColor == "red" ):
            print ( colored ( text = sTextMsg, color = sColor, attrs = [ "bold" ] ) )
        elif ( sColor == "green" ):
            print ( colored ( text = sTextMsg, color = sColor ) )
            
    return




