# -*- coding: utf-8 -*-
# Version vom 29. Juli 2026

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
# ************* Berechnung des zentrierten, gleitenden Durchschnittes mit der Möglichkeit des Padding auf die ursprüngliche Länge **********
#       Die Setzung tOrder = (0, n) führt zu einem vorausschauenden, die Setzung tOrder = (n, 0) zu einem rückblickenden Durchschnitt
def CalcRollingMean ( aData, tDegree, sPadMode = "edge", uStatLength = None ):
    CheckAssert ( bBool = ( sPadMode in [ "edge", "mean", "median", "symmetric", "none" ] ), sMsg = "Invalid Choice <sPadMode>!" )
    if ( sPadMode in [ "mean", "median" ] ):
        CheckAssert ( bBool = ( uStatLength is not None ), sMsg = "Invalid Value <uStatLength>!" )
        
    iLeft, iRight = tDegree[ 0 ], tDegree[ 1 ]
    aRange = np.arange ( start = iLeft, stop = aData.shape[ 0 ] - iRight + 1 )
    
    tRollMeanValues = tuple ( map ( lambda ik: np.nanmean ( aData[ ik - iLeft : ik + iRight ] ), aRange ) )
    aRollingMean = np.asarray ( tRollMeanValues, dtype = np.float64  )
    
    """
    ListRollMeanValues1 = list ()
    for ik in range ( iStart, aData.shape[ 0 ] - iEnd + 1 ):
        fRollMeanValue = np.mean ( aData[ ik - iStart : ik + iEnd ] )
        ListRollMeanValues1.append ( fRollMeanValue )
        
    aRollingMean1 = np.asarray ( ListRollMeanValues1, dtype = np.float64 )
    print ( np.allclose ( aRollingMean1, aRollingMean ) )
    """
    
    if ( sPadMode in [ "mean", "median" ] ):
        aRollingMean = np.pad ( array = aRollingMean, pad_width = ( tDegree[ 0 ], tDegree[ 1 ] - 1 ), mode = sPadMode, 
                                stat_length = uStatLength )
    elif ( sPadMode in [ "edge", "symmetric" ] ):
        aRollingMean = np.pad ( array = aRollingMean, pad_width = ( tDegree[ 0 ], tDegree[ 1 ] - 1 ), mode = sPadMode )
        
    return ( aRollingMean )
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
# ********************************* Detektion von Auffälligkeiten in einer Zeit abhängien Datenreihe ***************************************
def ScreenDataSeriesAbnormality ( aData, aDateTime = None, aReferenceData = None, fToleranceValue = None, iOutlierNumNeighbours = None, fThreshold = None ):
    if ( aDateTime is not None ):
        CheckAssert ( bBool = ( ( aDateTime.shape == aData.shape ) ), sMsg ="Inconsistent Shape of <aDateTime> and <aData>" )
        CheckAssert ( bBool = ( isinstance ( aDateTime[ -1 ], np.datetime64 ) ), sMsg = "Invalid Type <aDateTime>!" )
    
    if ( aReferenceData is not None ):
        CheckAssert ( bBool = ( ( aData.shape == aReferenceData.shape ) ), sMsg ="Inconsistent Shape of <aData> and <aReferenceData>!" )

    CheckAssert ( bBool = ( isinstance ( aData[ -1 ], ( np.int_, np.float32, np.float64 ) ) ), sMsg = "Invalid Type <aData>!" )
    
    #print ( ">> Shape Data: %s" % ( str ( aData.shape ) ) )
    
    DictResult = dict ()
    
    #### Anzahl der np.nan Werte  
    DictResult[ "NumberNA" ] = np.count_nonzero ( np.isnan ( aData ) )
    
    ### Anzahl der verschiedenen Werte
    aValues = np.unique ( ar = aData, return_counts = False )
    DictResult[ "NumberDifferentValues" ] = aValues.shape[ 0 ]
    
    #### Analyse der Zeitintervalle der aDateTimes
    if ( aDateTime is not None ):
        ListTimeDeltaSeconds = list ()
        for ik in range ( 1, aDateTime.shape[ 0 ] ):
            fTimeDelta = np.datetime64 ( aDateTime[ ik ] ) - np.datetime64 ( aDateTime[ ik - 1 ] )
            ListTimeDeltaSeconds.append ( fTimeDelta / np.timedelta64 ( 1, "s" ) )
    
        aTimeDeltaDiff = np.asarray ( ListTimeDeltaSeconds, dtype = np.float64 )
        aTimeDeltaQuartiles = CalcQuartiles ( aData = aTimeDeltaDiff, iAxis = 0, sInterpolationType = "linear" )
        DictResult[ "TimeDeltaQuartiles" ] = aTimeDeltaQuartiles 

    ### Zusammenhängende Plateaus von Werte oberhalb eines Schwellwerts werden verworfen
    if ( fThreshold is not None ):
        aThresholdIndices = np.arange ( start = 0, stop = aData.shape[ 0 ] )[ aData > fThreshold ]
        ## m benachbarte Indizes ( z.B. [ 12, 13, 14, 15 ] ) werden durch die Differenzbildung zu einer Sequenz von (m - 1) Einsen
        aCheck = np.diff ( a = aThresholdIndices, n = 1 ) 
        aThresholdSequenceLength = FindSequenceLength ( aArray = aCheck, iNumber = 1, bInfo = False )
        DictResult[ "IndicesThreshold" ] = aThresholdIndices
        DictResult[ "SequenceLengthThreshold" ] = aThresholdSequenceLength
    
    ### Bestimmung der Ausreißer als diejenigen Werte, die außerhalb des Bereiches [ aReferenz +/- fToleranzWert ] liegen
    if ( ( aReferenceData is not None ) and ( fToleranceValue is not None )  ):
        aOutlierSelect = np.logical_or ( aData < aReferenceData - fToleranceValue, aData > aReferenceData + fToleranceValue )
        aOutlierIndices = np.arange ( start = 0, stop = aData.shape[ 0 ] )[ aOutlierSelect ]

        if ( iOutlierNumNeighbours > 0 ):
            aOutlierSelect_extd = np.copy ( aOutlierSelect )
            for ik in range ( iOutlierNumNeighbours, aOutlierSelect.shape[ 0 ] - iOutlierNumNeighbours ):
                if ( aOutlierSelect[ ik ] == True ):
                    aOutlierSelect_extd[ ik - iOutlierNumNeighbours : ik + iOutlierNumNeighbours + 1 ] =  aOutlierSelect[ ik ]

            aOutlierIndices = np.arange ( start = 0, stop = aData.shape[ 0 ] )[ aOutlierSelect_extd ]
        
        aCheck = np.diff ( a = aOutlierIndices, n = 1 ) 
        aOutlierSequenceLength = FindSequenceLength ( aArray = aCheck, iNumber = 1, bInfo = False )
        DictResult[ "IndicesOutlier" ] = aOutlierIndices
        DictResult[ "SequenceLengthOutlier" ] = aOutlierSequenceLength
        
    return ( DictResult )
# ************************************************** Berechnung der Perzentile von Stichproben *********************************************
def CalcQuantiles ( aData, Quantiles, iAxis = 0, bRescale = True, sInterpolationType = "linear" ):
    CheckAssert ( bBool = ( isinstance ( Quantiles, ( tuple, list, np.ndarray, float ) ) ), sMsg = "Wrong Parameter Type!" )
        
    if ( bRescale == True ):     
        aData = RescaleData ( aData = aData, sType = "STANDARD", bReturnFullList = False )
        
    ## geändert 08. März 2026: if else Klausel herausgenommen
    #if ( aData.ndim == 1 ):
    aQ_obs = np.quantile ( a = aData, q = Quantiles, axis = iAxis, method = sInterpolationType )
    #else:
        # Vorsicht! Funktioniert nur mit Transposition 
        #aQ_obs = np.quantile ( a = aData, q = Quantiles, axis = iAxis, interpolation = sInterpolationType )
        #Q_obs1 = np.zeros ( ( Data.shape[ 1 ], len ( Quantiles ) ) )
        #for j in range ( 0, Data.shape[ 1 ] ):
         #   Q_obs[ j ] = np.quantile ( a = Data[ :, j ], q = Quantiles, interpolation = InterpolationType )
            
    return ( aQ_obs.T )
# **************************************************** Berechnung der Quartile von Stichproben *********************************************
def CalcQuartiles ( aData, iAxis = 0, sInterpolationType = "linear" ):
    aQuantiles = [ 0.0, 0.25, 0.5, 0.75, 1.0 ]
    
    aQ = CalcQuantiles ( aData = aData, Quantiles = aQuantiles, iAxis = iAxis, bRescale = False, sInterpolationType = sInterpolationType )
        
    return ( aQ )
# ********************************************************* Daten Reskalierung *************************************************************
# each row/line of matrix Data is a sample/observation and each column is a variable/feature 
def RescaleData ( aData, iAxis = 0, sType = "STANDARD", bReturnFullList = False ):
    sType = sType.upper ()
    CheckAssert ( bBool = ( sType in [ "CENTER_MEAN", "UNITY", "CENTER_MATCH_ZERO", "CENTER_ZERO", "STANDARD", "DIVIDE_STD" ] ),
                  sMsg = "Wrong Input Type!" )
    
    if ( iAxis is not None ):
        print ( ">> RescaleData > Using axis = %d for calculation!" % ( iAxis ) )
    else:
        print ( ">> RescaleData > Using flattened array for calculation!" )
        
    aMeanData = np.mean ( aData, axis = iAxis )
    aStdDevData = np.std ( aData, axis = iAxis )
    aMaxData = np.amax ( aData, axis = iAxis )
    aMinData = np.amin ( aData, axis = iAxis )

    aData_scaled = np.zeros ( shape = aData.shape, dtype = np.float64 )
    
    if ( sType == "CENTER_MEAN" ):
        aData_scaled = ( aData - aMeanData )
    # Bildet auf eine Teilmenge des Intervalls [-1, 1] ab, # wobei die 0 auf die 0 abgebildet wird
    elif ( sType == "CENTER_MATCH_ZERO" ): 
        print ( aMinData, aMaxData )
        aWidth = 2.0 * max ( np.abs ( aMaxData ), np.abs ( aMinData ) )
        aData_scaled = aData / aWidth 
    elif ( sType in [ "UNITY", "CENTER_ZERO" ] ):
        aWidth = ( aMaxData - aMinData )
        if ( np.any ( a = ( aWidth == 0.0 ) ) ):
        #if ( ( aWidth == 0.0 ).any () ):
            print ( "RescaleData > Adjusting [(Max - Min) == 0.0] Values!" )
            aWidth[ aWidth == 0.0 ] = 1.0 
        aData_scaled = ( aData - aMinData ) / aWidth 
        if ( sType == "CENTER_ZERO" ):
            aData_scaled -= 0.5
    elif ( sType in [ "STANDARD", "DIVIDE_STD" ] ):
        #if ( ( aStdDevData == 0.0 ).any () ):
        if ( np.any ( a = ( aStdDevData == 0.0 ) ) ):
            print ( "RescaleData > Adjusting [StdDev == 0.0] Values!" )
            aStdDevData[ aStdDevData == 0.0 ] = 1.0
        if ( sType == "STANDARD" ):
            aData_scaled = ( aData - aMeanData ) / aStdDevData
        else:
            aData_scaled = aData / aStdDevData
            
    if ( bReturnFullList == True ):
        return ( aData_scaled, aMeanData, aStdDevData, aMaxData, aMinData )
    else:    
        return ( aData_scaled )
# ************************* Detektion der Längen von Sequenzen ( m_1, m_2, m_3, ..., m_n) einer Zahl m in eimem Array ********************** 
def FindSequenceLength ( aArray, iNumber, bInfo = True ):
    ListGroupLength = list ()
    ListGroupElement = list ()
        
    for iKey, CGroup in groupby ( aArray ):
        ListGroupLength.append ( len ( list ( CGroup ) ) )
        ListGroupElement.append ( iKey )
            
    aGroupLength = np.asarray ( ListGroupLength, dtype = np.int16 )
    aGroupElement = np.asarray ( ListGroupElement, dtype =  np.int32 )
    
    if ( iNumber in aGroupElement ):
        aIndices = ( aGroupElement == iNumber )
        aResult = -1 * np.sort ( -aGroupLength[ aIndices ] )
        if ( bInfo == True ):
            print ( ">> FindSequenceLength > Found Sequence Length for Number %d" % ( iNumber ) )
            print ( aResult ) 
    else:
        aResult = None
        if ( bInfo == True ):
            print ( ">> FindSequenceLength > Number %d is not in Array!" % ( iNumber ) )
        
    return ( aResult )

