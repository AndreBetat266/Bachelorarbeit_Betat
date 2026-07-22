# -*- coding: utf-8 -*-
# Version vom 21. Juli 2026

import numpy as np
import os
import PlotBA as pl
from collections import OrderedDict
from inspect import stack
from termcolor import colored
from PIL import Image
from tabulate import tabulate

GiRandSeed = 17


# ********************* Universeller Assert-Handler  mit Angabe der aufrufenden Funktion, des Moduls und der Zeile *************************
def CheckAssert ( bBool, sMsg, sExtraInfo = None ):
    if ( bBool == False ):
        aS = stack ()
        sFileName = os.path.basename ( aS[ 1 ][ 1 ] )
        iLine = aS[ 1 ][ 2 ]
        sFunc = aS[ 1 ][ 3 ]
        sTextMessage = ">> Assertion Error: %s" % ( sFunc )
        print ( colored ( text = sTextMessage, color = "red", attrs = [ "bold" ] ) )
        
        sTextMessage = "\t > File: %s, Line: %d" % ( sFileName, iLine )
        print ( colored ( text = sTextMessage, color = "red", attrs = [ "bold" ] ) )
        
        if ( sExtraInfo is not None ):
            sTextMessage = "\t > %s [%s]" % ( sMsg, sExtraInfo )
        else:
            sTextMessage = "\t > %s" % ( sMsg )
        print ( colored ( text = sTextMessage, color = "red", attrs = [ "bold" ] ) )
        
        raise ( SystemExit )
    
    return 
# ******************************************* Sortierung der Items oder Keys eines Dictionary **********************************************
def SortDict ( Dict, bReverse = True ): # True : absteigend, False: aufsteigend
    DictSorted = OrderedDict ( sorted ( Dict.items (), key = lambda x: x[ 1 ], reverse = bReverse ) )

    return ( DictSorted )

# ********************************** Häufigkeits-Verteilung von kontinuierlichen Daten durch Diskretisierung *******************************
def CountFrequencyContinousData ( aData, iNumBins = 100, sReturnType = "RELATIVE", bCenterEdges = True, sColor = None, GraphicConfig = None, bInfo = True ):
    CheckAssert ( bBool = ( aData.ndim == 1 ), sMsg = "Data Shape must be 1-dimensional!" )
    
    sReturnType = sReturnType.upper ()
    CheckAssert ( bBool = ( sReturnType in [ "ABSOLUTE", "ABS", "RELATIVE", "REL", "NORMED", "NORM" ] ), sMsg = "Wrong Input Type!" )
    
    fMaxData = np.amax ( aData )
    fMinData = np.amin ( aData )
    fBinWidth = ( fMaxData - fMinData ) / iNumBins

    if ( sReturnType in [ "NORMED", "NORM" ] ):
        aCounts, aBinEdges = np.histogram ( a = aData, bins = iNumBins, density = True )
    else:
        aCounts, aBinEdges = np.histogram ( a = aData, bins = iNumBins, density = False )
        
     # reiner Check, nicht wirklich notwendig
    CheckAssert ( bBool = ( fBinWidth == 0.0 ) or ( np.allclose ( fBinWidth, aBinEdges[ -1 ]  - aBinEdges[ -2 ], atol = 1E-5 ) ),
                  sMsg = "Cross-Check failed!" )
    
    if ( sReturnType in [ "RELATIVE", "REL" ] ):
        aCounts = aCounts / np.sum ( aCounts )
        
    if ( bInfo == True ):
        print ( ">> CountFrequencyContinousData >\n\tMax: %.4f\n\tMin: %.4f\n\t#Bins: %d" % ( fMaxData, fMinData, iNumBins ) )
        print ( "\tBin-Width: %.4f\n\tSum (Bin-Width*Counts): %.4f" % ( fBinWidth, fBinWidth * np.sum( aCounts ) ) )
    
    if ( bCenterEdges == True ): # die insgesamt (NumBins + 1) Edges definieren die disjunkten, äquidistanten Intervallgrenzen    
        aBinEdges = aBinEdges[ : -1 ] + 0.5 * fBinWidth
    else:
        aBinEdges = aBinEdges[ : -1 ]
        
    if ( sColor is not None ):
        if ( GraphicConfig is None ):
            if ( sReturnType in [ "RELATIVE", "REL" ] ):
                sLabelTextY = "Relative Häufgigkeit $H_{rel}$"
            elif ( sReturnType in [ "ABSOLUTE", "ABS" ] ):
                sLabelTextY = "Absolute Häufigkeit $H_{abs}$"
            elif ( sReturnType in [ "NORMED", "NORM" ] ):
                sLabelTextY = "normierte Häufigkeit $H_{norm}$"
            GraphicConfig = pl.CGraphicConfig ( sTitle = "Histogramm reeller, eindimensionaler Daten", sLabelX = "Wert $x_k$", sLabelY = sLabelTextY )
            
        fBarWidth = np.around ( ( fBinWidth ), decimals = 2 )

        pl.PlotBarChart ( aX = aBinEdges, aData = aCounts, GraphicConfig = GraphicConfig, uColor = sColor, sEdgeColor = "black", fWidth = fBarWidth )
    
    return ( aCounts, aBinEdges, fBinWidth )
# **************************** Skalierung der Größe eines Bildes aus einer Datei um einen ganzzahligen Faktor ******************************
def RescaleImageFile ( sPathImageFile, iScaleFactor, bReturnImagePIL ):  
    assert ( type ( iScaleFactor ) == int )
    
    ImagePIL = Image.open ( sPathImageFile )
    iActualWidth, iActualHeight = ImagePIL.size
    iNewWidth, iNewHeight = iActualWidth // iScaleFactor, iActualHeight // iScaleFactor
    
    RescaledImagePIL = ImagePIL.resize ( ( iNewWidth, iNewHeight ), resample = Image.Resampling.LANCZOS, reducing_gap = 4.0 )
        
    if ( bReturnImagePIL == False ):
        return ( np.array ( RescaledImagePIL ) )
    else:    
        return ( RescaledImagePIL )
# ************************* Umwandlung eines RGB-Bildes, welches als Numpy-Array vorliegt, in ein Schwarz-Weiß-Bild ************************
# ********************** hierbei werden die drei Farb-Känale R, G, B surjektiv auf das Intervall [ 0, 1 ] abgebildet ***********************
def ConvertRGBImageArrayToGray ( aImageArray, iMaxColors = None, bConvert3D = False ):
    iMax_ubyte = np.iinfo ( np.ubyte ).max # 255
    iMax_ushort = np.iinfo ( np.ushort ).max
    
    assert ( aImageArray.ndim == 3 )
    if ( iMaxColors is not None ):
        assert ( 2 <= iMaxColors <= iMax_ushort )

    # The sRGB color space is defined in terms of the CIE 1931 linear luminance Ylinear, which is given by
    # Y_linear = 0.2126 * R_linear + 0.7152 * G_linear + 0.0722 * B_linear
    aConv_CIE1931 = np.asarray ( [ 0.2126, 0.7152, 0.0722 ], dtype = np.float64 ) # Konversion gemäß CIE 1931
    #aConv_skimage = np.asarray ( [ 0.2125, 0.7154, 0.0721 ], dtype = np.float32 ) # Konversion nach der Library SkImage
    # aConv_pillow = np.asarray ( [ 0.299, 0.587, 0.114 ], dtype = np.float32 ) # Konversion nach der Library PILLOW
    
    aImageArrayGray = np.matmul ( aImageArray, np.transpose ( aConv_CIE1931 ) ) / float ( iMax_ubyte )
    
    #print ( np.amax ( aImageArrayGray ), np.amin ( aImageArrayGray ) )
    
    if ( iMaxColors is not None ):
        if ( ( iMaxColors - 1 ) <= iMax_ubyte ): # geändert 08.02.2026
            aImageArrayGray = np.ubyte ( iMaxColors * aImageArrayGray ) 
            
        #if ( iMaxColors <= iMax_ubyte ):
        #    aImageArrayGray = np.ubyte ( ( iMaxColors - 1 ) * aImageArrayGray ) # -1 , da die 0 als Wert mitzählt
        else:
            aImageArrayGray = np.ushort ( ( iMaxColors - 1 ) * aImageArrayGray ) # -1 , da die 0 als Wert mitzählt
        
    if ( bConvert3D == True ):
        aImageArrayGray = np.reshape ( aImageArrayGray, shape = ( aImageArrayGray.shape[ 0 ], aImageArrayGray.shape[ 1 ], 1 ) )
        aImageArrayGray = np.repeat ( aImageArrayGray, 3, axis = 2 )
    
    return ( aImageArrayGray )
# ************************* Funktion, die aus einem 2-dimensionalen Array (z.B einem Bild) eine Stichprobe zieht ***************************
def SampleFromData2D ( aData2Dim, fRatio, fEmptyValue, sColorMap = "Grays_r", iRandSeed = GiRandSeed ):
    CheckAssert ( bBool = ( ( isinstance ( aData2Dim, np.ndarray ) ) and ( aData2Dim.ndim == 2 ) ), sMsg = "Invalid Type <aData2Dim>!" )
    iSize = aData2Dim.size
    aSampleData2D = np.full ( shape = aData2Dim.shape, fill_value = fEmptyValue, dtype = np.float32 )
    
    iNumObservations = int ( np.floor ( fRatio * float ( iSize ) ) )
    CRnG = np.random.default_rng ( seed = iRandSeed )
    aSamples = CRnG.choice ( iSize, size = iNumObservations, replace = False )
    
    aCoordsX, aCoordsY = np.divmod ( aSamples, aData2Dim.shape[ 1 ] )
    tSampleObservations = tuple ( map ( lambda tCoord : aData2Dim[ tCoord[ 0 ], tCoord[ 1 ] ], zip ( aCoordsX, aCoordsY ) ) )
    aSampleData2D[ aCoordsX, aCoordsY ] = aData2Dim[ aCoordsX, aCoordsY ]
        
    if ( sColorMap is not None ):
        CGraCon = pl.CGraphicConfig ( sTitle = "Beobachtung (%.0f%% des Originalbilds)" % ( 100.0 * fRatio ), sLabelX = "x", sLabelY = "y", sLegend = "Grauwert" )
        pl.PlotImage ( aData2Dim = aSampleData2D, GraphicConfig = CGraCon, sColorMap = sColorMap, sInterpolation = "spline36", 
                       sOrigin = "upper", tExtent = None, sGridAxis = "both" )
        
    return ( aSampleData2D, ( aCoordsX, aCoordsY ), tSampleObservations )
# ******** Analyse einer beliebigen Datei bzgl. Datentyp der Spalten, fehlender Werte, Median, arith. Mittelwert und #Ausprägungen *********
# +++++++++++++++++++++++++++++++++++++++++++++++ Auslesen der Kommentar-Zeile aus einer Datei +++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++ Zurückgegeben wird der LETZTE der Eintrage, die als Kommentar gekennzeichnet sind !
def ScanFileForComment ( sFileName, sComments, sDelimiter, sStrip = "\"\'" ):
    ListDescriptions = list ()
    CFile = open ( file = sFileName, encoding = "latin-1" )
   
    for ik in range ( 1, 10 ):
        sLine = CFile.readline ()
        sLine = sLine.strip ( "\n" )
        if ( sComments is None ):
            ListDescriptions.append ( sLine.split ( sDelimiter ) )
            break
        else:    
            iIndex = sLine.find ( sComments )
            if ( iIndex != -1 ):
                sHeader = sLine.replace ( sComments, "" )
                if ( sStrip is not None ):
                    for iChar in sStrip:
                        sHeader = sHeader.replace ( iChar, "" )
                ListDescriptions.append ( sHeader.split ( sDelimiter ) )
    
    iNumComments = len ( ListDescriptions )
    
    if ( iNumComments == 0 ):
        print ( colored ( text = ">> ScanFileForComment > No Comments Found!", color = "magenta", attrs = [ "bold" ] ) )
    elif ( iNumComments > 1 ):
        print ( colored ( text = ">> ScanFileForComment > Found %d Comment Lines!" % ( iNumComments ), color = "green" ) )
    
    return ( np.asarray ( ListDescriptions[ -1 ] ) )
# ********************* Konvertiert ein Array von Zahlen von deutscher in englsiche Dezimzaltrennung und umgekehrt ************************
def ConvertNumericArray ( aArray, sTargetType ):
    sTargetType = sTargetType.upper ()
    CheckAssert ( bBool = ( sTargetType in [ "DE", "EN" ] ), sMsg = "Wrong Parameter for <sTargetType>!" )
    CheckAssert ( bBool = ( isinstance ( aArray, np.ndarray ) ), sMsg = "Input must be an Array!" )
    
    if ( aArray.dtype.type != np.str_ ):
        aArray = np.asarray ( a = aArray, dtype = str )
    
    if ( sTargetType == "EN" ):
        aArray = np.char.replace ( a = aArray, old = ".", new = "" )
        aArray = np.char.replace ( a = aArray, old = ",", new = "." )
    elif ( sTargetType == "DE" ):
        aArray = np.char.replace ( a = aArray, old = ",", new = "" )
        aArray = np.char.replace ( a = aArray, old = ".", new = "," )
    
    
    return ( aArray )
# +++++++++++++++++++++++++++++++++++++++++++++++++ Struktur Analyse einer Daten-Datei +++++++++++++++++++++++++++++++++++++++++++++++++++++
def AnalyzeDataStructure ( sDataFileName, tIgnoreColumn = None, tGermanFormatColumn = None, sComments = "#", sDelimiter = "\t", 
                           bAddSummary = True, bAddInfoNA = True, bReturnCheckList = False ):
    aDescriptions = ScanFileForComment ( sFileName = sDataFileName, sComments = sComments, sDelimiter = sDelimiter )
    if ( sComments is None ): # keine Kommentar-Zeile mit Header
        aRawData = np.loadtxt ( fname = sDataFileName, comments = sComments, skiprows = 1, delimiter = sDelimiter, encoding = "latin-1", dtype = str ) 
    else:
        aRawData = np.loadtxt ( fname = sDataFileName, comments = sComments, delimiter = sDelimiter, encoding = "latin-1", dtype = str )  
    iMaxLengthModusValue = 10
    
    aRawData = np.char.strip ( a = aRawData, chars = "\"" )
    #print ( aRawData.shape )
    aIndex = np.arange ( start = 0, stop = aRawData.shape[ 1 ] )
    
    if ( tIgnoreColumn is not None ):
        CheckAssert ( bBool = ( isinstance ( tIgnoreColumn, ( list, tuple, np.ndarray ) ) ), sMsg = "Wrong Input Type!" ) 
        aIndex = np.delete ( arr = aIndex, obj = tIgnoreColumn )

    aDescriptions = aDescriptions[ aIndex ]
    aData = aRawData[ :, aIndex ]
    if ( bAddSummary == True ):
        print ( ">> AnalyzeDataStructure > Loaded Data of shape: %s" % ( str ( aData.shape ) ) )
        
    ListCheck = list ()
    ListData = list ()
    
    for iSpalte in range ( aData.shape[ 1 ] ): 
        aSpalte = aData[ :, iSpalte  ]
    
        aIndicesEmpty = ( aSpalte == "" )
        iSumIndicesEmpty = np.sum ( aIndicesEmpty )
        aSpalte[ aIndicesEmpty ] = np.nan
   
        aIndicesNA = ( aSpalte == "NA" )
        iSumIndicesNA = np.sum ( aIndicesNA )
        aSpalte[ aIndicesNA ] = np.nan
        if ( bAddInfoNA == True ):
            print ( ">> Spalte: %d, empty: %d, NA: %d" % ( iSpalte, iSumIndicesEmpty, iSumIndicesNA ) )

        aValues, aCounts = np.unique ( ar = aSpalte, return_counts = True )
        iMaxIndex = np.argmax ( a = aCounts )
        sModusValue = aValues[ iMaxIndex ]
        if ( len ( sModusValue ) > iMaxLengthModusValue ):
            sModusValue = sModusValue[ : iMaxLengthModusValue - 3 ] + "..."     
        sModus = "%s (%s)" % ( sModusValue, aCounts[ iMaxIndex ] )
        
        fMean = np.nan
        fMedian = np.nan
        fMax = np.nan
        fMin = np.nan
        aDatenSpalte = aSpalte
        
        try:
            sDatenTyp = "float"
            if ( tGermanFormatColumn is not None ):
                if ( iSpalte in tGermanFormatColumn ):
                    aSpalte = ConvertNumericArray ( aArray = aSpalte, sTargetType = "EN" ) ### zum Handling des deutschen Dezimaltrenner Formats
        
            aDatenSpalte = np.asarray ( list ( map ( float, aSpalte ) ), dtype = np.float64 ) 
            fMean = np.round ( a = np.nanmean ( aDatenSpalte ), decimals = 2 )
            fMedian = np.round ( a = np.nanmedian ( aDatenSpalte ), decimals = 2 )
            fMax = np.round ( a = np.nanmax ( aDatenSpalte ), decimals = 2 )
            fMin = np.round ( a = np.nanmin ( aDatenSpalte ), decimals = 2 )
        except ValueError:
            sDatenTyp = "string"
            
        try:
            sPrevDatenTyp = sDatenTyp
            sDatenTyp = "int"
            aDatenSpalte = np.asarray ( list ( map ( int, aSpalte ) ), dtype = np.int64 ) 
            fMean = np.round ( a = np.nanmean ( aDatenSpalte ), decimals = 2 )
            fMedian = np.round ( a = np.nanmedian ( aDatenSpalte ), decimals = 2 )
            fMax = np.round ( a = np.nanmax ( aDatenSpalte ), decimals = 2 )
            fMin = np.round ( a = np.nanmin ( aDatenSpalte ), decimals = 2 )
        except ValueError:
            sDatenTyp = sPrevDatenTyp    
                
        sRange = "[%.0f,%.0f]" % ( fMin, fMax )
        if ( aDescriptions is not None ):
            sDescription = aDescriptions[ iSpalte ]
        else:
            sDescription = "%d" % ( iSpalte )
        
        #if ( bAddSummary == True ):
        ListCheck.append ( ( sDescription, sDatenTyp, iSumIndicesEmpty + iSumIndicesNA, fMean, fMedian, sRange, sModus, aValues.shape[ 0 ] ) )
        ListData.append ( aDatenSpalte )
        
    if ( bAddSummary == True ):
        tHeader = [ "Column", "Type", "#NA", "Mean", "Median", "Span", "Mode", "#Classes"]
        print ( tabulate ( tabular_data = ListCheck, headers = tHeader, tablefmt = "pretty" ) )
        #print ( tabulate ( tabular_data = ListCheck, headers = tHeader, tablefmt = "latex" ) )
        
    if ( bReturnCheckList == True ):
        return ( aDescriptions, ListCheck, ListData )
    else:
        return ( aDescriptions, ListData )
