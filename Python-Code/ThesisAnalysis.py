# -*- coding: utf-8 -*-
# Version vom 29. Juli 2026

import numpy as np
import matplotlib.pyplot as plt 
import PlotBA as pl
import GeodataAnalysisBA as ga
import SpatialAnalysisBA as sa
import DataImportSensorBA as ds
from cartopy import crs
from scipy.interpolate import CubicSpline
from UtilitiesBA import CheckAssert, SortDict, CalcRollingMean, ScreenDataSeriesAbnormality


fLon_ref, fLat_ref = ds.GDictConfig.get ( "CentralLocation" )
CCRS_azmequi = crs.AzimuthalEquidistant ( central_longitude = fLon_ref, central_latitude = fLat_ref )


# ***************** Leave One Out Cross Validation Ansatz zum Austesten der Güte von anisotropen Kovarianz Modellen ************************
def RunAnisotropicCrossValidation ( sModel, tParameterLambda, tParameterVar, iNumStepsTheta, sDate, sSubFolderMonthYear, 
                                    sDataSelection, fVar_fix = 4.0, fLambda_fix = 4.0, sAverageParameter = "median" ):
    
    iSampleRun = 34
    fDegTheta_fix =  60.0
    CheckAssert ( bBool = ( len ( tParameterLambda ) == 3 and tParameterLambda[ - 1 ] == 4 ), sMsg = "Invalid Format <tParameterLambda>!" )
    CheckAssert ( bBool = ( len ( tParameterVar ) == 3 and tParameterVar[ - 1 ] == 3 ), sMsg = "Invalid Format <tParameterVar>!" )
    
    fLenScaleX = 21300.0
    fNugget = 0.2
    aSelectIndices, tLabel, sDataType = ga.GDictDataColumns.get ( sDataSelection )

    aRawData, DictStatistic = ds.LoadDataStatisticMonth ( sDataType = sDataType, sSubFolderMonthYear = sSubFolderMonthYear, 
                                                          sAverageParameter = sAverageParameter )
    """
    ga.ShowDataSnapShotwGraph ( aRawData = aRawData, sDate = sDate, sDataSelection = sDataSelection, CProjCRS = CCRS_azmequi, 
                                bShowDistribution = False )
    ga.ShowDataSnapshotwBorder ( aRawData = aRawData, sDataSelection = sDataSelection, sDate = sDate, CProjCRS = CCRS_azmequi, 
                                 tStyleRectangle = None, bShowDistribution = False )
    """
    aData = np.asarray ( aRawData[ :, aSelectIndices ], dtype = np.float64 )
    
    fMinX, fMaxX = np.amin ( aData[ :, 0 ] ), np.amax ( aData[ :, 0 ] )
    fMinY, fMaxY = np.amin(  aData[ :, 1 ] ), np.amax ( aData[ :, 1 ] ) 
    
    iArgMax = np.argmax ( aData[ :, 2 ] )
    aDataID = np.arange ( start = 1, stop = len ( DictStatistic ) + 1 )
    DictStatistic_inv = { tTupel[ 0 ][ 0 ]: sKey for sKey, tTupel in DictStatistic.items () }
    sSensorID_max = DictStatistic_inv.get ( aDataID[ iArgMax ] )
    print ( ">> EstimateDirectionalVariogram > Max: %.2f (ID: %s)" % ( aData[ iArgMax][ 2 ], sSensorID_max ) )
    #ShowDataSingleID ( sSensorID = sSensorID_max, tSequence = ( "2025-11-01", "2025-11-02" ), sDataSelection = sDataSelection, 
    #                   sSubFolderMonthYear = sSubFolderMonthYear )

    #ga.ShowDistributionMeasurements ( aData = aData[ :, 2 ], sTitleStartText = sDate, sDescription = tLabel[ 2 ], sUnit = tLabel[ 3 ], 
     #                                iNumBins = 50 )
     
     
    fLambdaStart, fLambdaEnd, iNumStepsLambda = tParameterLambda
    fVarStart, fVarEnd, iNumStepsVar = tParameterVar
    
    aRadTheta = np.linspace ( start = 0.0, stop = 2.0 * np.pi, num = iNumStepsTheta, endpoint = False )
    aLambda = np.linspace ( start = fLambdaStart, stop = fLambdaEnd, num = iNumStepsLambda, endpoint = True )
    aVar = np.linspace ( start = fVarStart, stop = fVarEnd, num = iNumStepsVar, endpoint = True )
    CheckAssert ( bBool = ( fVar_fix in aVar and fLambda_fix in aLambda ), sMsg = "Invalid Parameter Choice" )
    
    aDegTheta = np.rad2deg ( aRadTheta )
    ListCV = list ()
    ListMSE = list ()

    for fVar in aVar:
        for fLambda in aLambda:
            for fRadTheta, fDegTheta in zip ( aRadTheta, aDegTheta ):
                uLenScale = ( fLenScaleX, fLenScaleX * fLambda )
                CCovModel = sa.CCovarianceModelGsT ( sModel = sModel, fVar = fVar, uLenScale = uLenScale, fNugget = fNugget, 
                                                     fRescale = 1.0, fAngle = fRadTheta, fShape = 2.0, bInfo = False )
                CKrige = sa.COrdKrigingGsT ( CCovarianceModel = CCovModel, uDataObserved = np.transpose ( aData ), bFitVariogram = False )
                
                if ( ( np.allclose ( fDegTheta, fDegTheta_fix ) ) and ( fVar == fVar_fix ) and ( fLambda == fLambda_fix ) ):
                    ### warum muss hier der Winkel um 90 Grad gedreht werden ?? Wegen der Transposition der Daten ??
                    fRadTheta_corrected = fRadTheta #- np.pi / 2.0
                    CCovModel2 = sa.CCovarianceModelGsT ( sModel = sModel, fVar = fVar, uLenScale = uLenScale, fNugget = fNugget, 
                                                         fRescale = 1.0, fAngle = fRadTheta_corrected, fShape = 2.0, bInfo = False )
                    CKrige2 = sa.COrdKrigingGsT ( CCovarianceModel = CCovModel2, uDataObserved = np.transpose ( aData ), bFitVariogram = False )
                    
                    CKrige2.Interpolate ( tDimX = ( fMinX, fMaxX, 800 ), tDimY = ( fMinY, fMaxY, 800 ), tLimX = ( 300, 300 ), tLimY = ( 300, 300 ), 
                                          iNumLevel = 80, sColorMap = "RdYlBu_r" )
                
                    #fCV_MSE = CKrige2.Predict ( uPos = [ aData[ 0, : ], aData[ 1, : ] ], aDataObserved = aData[ 2, : ] )
                    #print ( "Fehler: %.3f" % ( fCV_MSE ) )
                    
                aCV_MSE = CKrige.RunCrossValidation ( iNumFolds = aData.shape[ 0 ], bShowInfo = False )
                ListMSE.append ( aCV_MSE )
                ListCV.append ( ( fVar, fLambda, fDegTheta, np.mean ( aCV_MSE ) ) )

    aResultCV = np.asarray ( ListCV )
    aResultMSE = np.asarray ( ListMSE )    
    
    ### Ein beispielhfater Ausruck des Verlaufs für einen Durchgang mit Mittelwert und Standardabweichung
    aCV_MSE_single = aResultMSE[ iSampleRun ]
    CGraCon = pl.CGraphicConfig ( sTitle = "Beispiel eines LOO-CV Durchlaufs; %s" % ( sDate ), sLabelX = "Nummer des Sensors in der Test-Menge",
                                 sLabelY = "$\log\,(e_j)$" )
    
    fLogMean = np.log ( np.mean ( aCV_MSE_single ) )
    fLogStd = np.log ( np.std ( aCV_MSE_single ) )
    CHLine1 = pl.CLine ( sLineColor = "o12", fLinePos = fLogStd, fLineWidth = 2.0, sLineStyle = "--", sLineLabel = "Stdabw. der $e_j$" )
    CHLine2 = pl.CLine ( sLineColor = "r12", fLinePos = fLogMean, fLineWidth = 2.0, sLineStyle = "-.", sLineLabel = "Mittelwert der $e_j$" )
    CGraCon.Set ( HLine1 = CHLine1, HLine2 = CHLine2 )
    pl.PlotXY ( aX = np.arange ( start = 1, stop = aCV_MSE_single.shape[ 0 ] + 1 ), aY = np.log ( aCV_MSE_single ), 
                tStyle = ( "b10", "o", 7.0, "--", 2.0, "" ), GraphicConfig = CGraCon )
    
    ListY = list ()
    ListLegend = list ()

    for fVar in aVar:
        sVar = "Sill: %.1f" % ( fVar )
        ListLegend.append ( sVar )
        aSelectIndices = np.logical_and ( aResultCV[ : , 0 ] == fVar, aResultCV[ : , 1 ] == fLambda_fix )
        aResult = aResultCV[ aSelectIndices ]
        ListY.append ( aResult[ :, -1 ] )
        
    CGraCon = pl.CGraphicConfig ( sTitle = "LOO-CV Ergebnisse der %s Messungen von %s ($\\lambda=%.1f$)" % ( tLabel[ 2 ], sDate, fLambda_fix ), 
                                  sLabelX = "Rotationswinkel $\\vartheta$ (Grad)", sLabelY = "$e_{\mathrm{MSE}}$" )
    
    pl.PlotX3Y ( aX = aResult[ :, 2 ], aY1 = ListY[ 0 ], aY2 = ListY[ 1 ], aY3 = ListY[ 2 ], 
                tStyleY1 = ( "o10", "o", 8.0, "--", 2.0, ListLegend[ 0 ] ),
                tStyleY2 = ( "b10", "o", 8.0, "--", 2.0, ListLegend[ 1 ] ),
                tStyleY3 = ( "s10", "o", 8.0, "--", 2.0, ListLegend[ 2 ] ), GraphicConfig = CGraCon )   


    ListY = list ()
    ListLegend = list ()
    for fLambda in aLambda:
        if ( fLambda == 1.0 ):
            sLambda = "isotrop"
        else:
            sLambda = "$\\lambda: %.1f$" % ( fLambda )

        ListLegend.append ( sLambda )
        aSelectIndices = np.logical_and ( aResultCV[ : , 0 ] == fVar_fix, aResultCV[ : , 1 ] == fLambda )
        aResult = aResultCV[ aSelectIndices ]
        ListY.append ( aResult[ :, -1 ] )
        
    CGraCon.Set ( sTitle = "LOO-CV Ergebnisse der %s Messungen von %s ($\mathrm{Sill}=%.1f$)" % ( tLabel[ 2 ], sDate, fVar_fix ) ) 
    pl.PlotX4Y ( aX = aResult[ :, 2 ], tY = ListY, 
                 tStylesY = ( ( "g10", "o", 8.0, "-", 2.0, ListLegend[ 0 ] ),
                             ( "o10", "o", 8.0, ":", 2.0, ListLegend[ 1 ] ),
                             ( "b10", "o", 8.0, "--", 2.0, ListLegend[ 2 ] ), 
                             ( "s10", "o", 8.0, "-.", 2.0, ListLegend[ 3 ] ) ), GraphicConfig = CGraCon )   
    
    
    aSelectIndices = np.logical_and ( aResultCV[ : , 0 ] == fVar_fix, aResultCV[ : , 1 ] == 1.0 )
    aResult_iso = aResultCV[ aSelectIndices ][ :, -1 ]
    
    ListD = list ()
    for fLambda in aLambda[ 1 : ]:
        aSelectIndices = np.logical_and ( aResultCV[ : , 0 ] == fVar_fix, aResultCV[ : , 1 ] == fLambda )
        aResult_aniso = aResultCV[ aSelectIndices ][ :, -1 ]
        ListD.append ( ( aResult_iso - aResult_aniso ) )
        
    aD = np.asarray ( ListD, dtype = np.float64 )
    print ( aD.shape )
    aMean_D = np.mean ( aD, axis = 1 )
    aS_D = np.std ( aD, ddof = 1, axis = 1  )
    print ( aMean_D )
    print ( aS_D )
    
    print ( aMean_D / aS_D )

    
    """
    pl.PlotX2Y ( aX = aResultCV[ :, 2 ], aY1 = aResultCV[ :, 3 ], aY2 = aResultCV[ :, -1  ], 
                 tStyleY1 = ( "o10", "o", 8.0, "--", 2.0, "Std" ),
                 tStyleY2 = ( "b10", "o", 8.0, "--", 2.0, "Mean" ),
                 GraphicConfig = CGraCon )   
    """
    return

# ******************************* Analyse der direktionalen Variogramm für verschiedenen Zeiträume *****************************************
def ShowDirectionalVariogram ( sDataSelection, sAverageParameter = "median", sMonthYear = None, bShowFit = True ):
    iNumDirections = 18
    sModel = "matern"  
    bUseNugget = True
    
    if ( sMonthYear is None ):
        tMonthYear = ( "Jul2025", "Nov2025", "Apr2026" )
        #tMonthYear = ( "Apr2026", )
    else:
        tMonthYear = ( sMonthYear, )

    aSelectIndices, tLabel, sDataType = ga.GDictDataColumns.get ( sDataSelection )
    
    for sMonthYear in tMonthYear:
        aRawData, DictStatistic = ds.LoadDataStatisticMonth ( sDataType = sDataType, sSubFolderMonthYear = sMonthYear, 
                                                              sAverageParameter = sAverageParameter )

        aDataID = np.arange ( start = 1, stop = len ( DictStatistic ) + 1 )
        #DictStatistic_inv = { tTupel[ 0 ][ 0 ]: sKey for sKey, tTupel in DictStatistic.items () }
        aData_sel = np.asarray ( aRawData[ :, aSelectIndices ], dtype = np.float64 )
        iArgMax = np.argmax ( aData_sel[ :, 2 ] )
        #   sSensorID_max = DictStatistic_inv.get ( aDataID[ iArgMax ] )
        #   print ( ">> EstimateDirectionalVariogram > Max: %.2f (ID: %s)" % ( fMax, sSensorID_max ) )
        #   ta.ShowDataSingleID ( sSensorID = sSensorID_max, tSequence = ( "2026-04-01", "2026-04-30" ), sDataSelection = sDataSelection )

        #ga.ShowDistributionDistances ( aData = aData_sel, sDescription = tLabel[ 2 ], sUnit = tLabel[ 3 ] )
        
        aCoords = np.asarray ( aData_sel[ :, : 2 ], dtype = np.float64 )
        aData = np.asarray ( aData_sel[ :, 2 ], dtype = np.float64 ) 

        ### ANFANG: Entferung des Maximum der Sensoren !
        aSelect = ( aDataID != aDataID[ iArgMax ] )
        aCoords = aCoords[ aSelect, : ]
        aData = aData[ aSelect ]
        ### ENDE: Entferung des Maximum der Sensoren !

        sDate = sMonthYear[ : 3 ] + " " + sMonthYear[ 3 : ]
        #sAverageParameter, sDescription, sUnit, iNumSensors = tTextLabel
        if ( bShowFit == True ):
            tTextLabel = ( sAverageParameter, tLabel[ 2 ], tLabel[ 3 ], len ( DictStatistic ) - 1 )
        else:
            tTextLabel = None
        sa.EstimateDirectionalVariogram ( aCoords = aCoords, aData = aData, sModel = sModel, sDate = sDate, iNumDirections = iNumDirections, 
                                          sPivotParameter = "sill", tTextLabel = tTextLabel, bUseNugget = bUseNugget )
        
        #sa.EstimateDirectionalVariogram ( aCoords = aCoords, aData = aData, sModel = sModel, sDate = sDate, iNumDirections = iNumDirections, 
         #                                 sPivotParameter = "nugget", tTextLabel = None, bUseNugget = bUseNugget )

        
        sa.EstimateDirectionalVariogram ( aCoords = aCoords, aData = aData, sModel = sModel, sDate = sDate, iNumDirections = iNumDirections, 
                                          sPivotParameter = "range", tTextLabel = None, bUseNugget = bUseNugget )
        
    return

# *********************** Zusammenstellen der Daten für Mittelwert/Median über eine Zeitspanne und einer eventuellen ***********************
###                            Eingrenzung der Sensoren anhand der Lage in einem umschreibenden Rechteck
def ShowDataTimeAverage ( sDate, sDataSelection, tParameterEstimator,  sAverageParameter = "median", tBorderX = None, tBorderY = None ):   
    aSelectIndices, tLabel, sDataType = ga.GDictDataColumns.get ( sDataSelection )
    
    aRawData, DictStatistic = ds.LoadDataStatisticMonth ( sDataType = sDataType, sAverageParameter = sAverageParameter )
    aDataID = np.arange ( start = 1, stop = len ( DictStatistic ) + 1 )
    
    #aRawData, aDataID, DictInfo = dc.LoadDataTimeFrame ( sDate = sDate, sStartTime = sStartTime, sDataType = sDataType, iWindowDelta = iWindowDelta, 
    #                                                     bUseMeanMatches = bUseMeanMatches )
    
    DictStatistic_inv = { tTupel[ 0 ][ 0 ]: sKey for sKey, tTupel in DictStatistic.items () }
    aData_sel = aRawData[ :, aSelectIndices ]
    
    
    if ( tBorderX is not None and tBorderY is not None ):
        tStyleRectangle = ( tBorderX[ 0 ], tBorderY[ 0 ], tBorderX[ 1 ] - tBorderX[ 0 ], tBorderY[ 1 ] - tBorderY[ 0 ], 2.0, "-.", "r12", "none" )

        aSelectX = np.logical_and ( aData_sel[ :, 1 ] >= tBorderX[ 0 ], aData_sel[ :, 1 ] <= tBorderX[ 1 ] )
        aSelectY = np.logical_and ( aData_sel[ :, 0 ] >= tBorderY[ 0 ], aData_sel[ :, 0 ] <= tBorderY[ 1 ] )
        aSelect = np.logical_and ( aSelectX, aSelectY )
        print ( ">> Found Sensors: %d" % ( aSelect.shape ) )
        aRawData = aRawData[ aSelect, : ]
        aDataID = aDataID[ aSelect ]
        print ( ">> New Data Shape: %s" % ( str ( aRawData.shape ) ) )
        aData_sel = aRawData[ :, aSelectIndices ]
    else:
        tStyleRectangle = None
    
    fMinX, fMaxX = np.amin ( aData_sel[ :, 0 ] ), np.amax ( aData_sel[ :, 0 ] )
    fMinY, fMaxY = np.amin(  aData_sel[ :, 1 ] ), np.amax ( aData_sel[ :, 1 ] ) 
    
    ga.ShowDataSnapshotwBorder ( aRawData = aRawData, sDataSelection = sDataSelection, sDate = sDate, CProjCRS = CCRS_azmequi, 
                                 tStyleRectangle = tStyleRectangle, bShowDistribution = False )

    print ( ">> Shape Data_sel: %s" % ( str ( aData_sel.shape ) ) )
    
    fMax, iArgMax = np.amax ( aData_sel[ :, 2 ] ), np.argmax ( aData_sel[ :, 2 ] )
    sSensorID_max = DictStatistic_inv.get ( aDataID[ iArgMax ] )
    print ( ">> ShowDataTimeAverage > Max: %.2f (ID: %s)" % ( fMax, sSensorID_max ) )
    ShowDataSingleID ( sSensorID = sSensorID_max, tSequence = ( "2026-04-01", "2026-04-30" ), sDataSelection = sDataSelection )
    
    ga.ShowDistributionDistances ( aData = aData_sel, sDescription = tLabel[ 2 ], sUnit = tLabel[ 3 ] )
    
    CVario = sa.CVariogramSkG ( aData = aData_sel, sEstimator = "cressie", bUseNugget = True, iNumLags = 15, sColorMap = None )
    CVario.ShowVariogramEstimation ( tParameterEstimator = tParameterEstimator )
    
    aRank, ListRankEstimator = CVario.CompareCovModel ()
    CVario.CheckIsotropy ( sEstimator = "cressie", sModel = ListRankEstimator[ 0 ], iNumSamplePoints = None, uMaxLag = 8000, iNumLags = 14, 
                           sBinFunc = "even" )
    
    print ( ListRankEstimator[ 0 ], aRank[ 0 ] )
    fRange, fSill, fNugget = aRank[ 0 ][ 1 ], aRank[ 0 ][ 2 ], aRank[ 0 ][ 3 ]
    CCovModel = sa.CCovarianceModelGsT ( sModel = ListRankEstimator[ 0 ], fVar = ( fSill - fNugget ), uLenScale = fRange, fNu = 1.5, fAlpha = 1.0, fNugget = fNugget )
    CKrige = sa.COrdKrigingGsT ( CCovarianceModel = CCovModel, uDataObserved = np.transpose ( aData_sel ), bFitVariogram = True )
    CKrige.Interpolate ( tDimX = ( fMinX, fMaxX, 400 ), tDimY = ( fMinY, fMaxY, 400 ), tLimX = ( 300, 300 ), tLimY = ( 300, 300 ), 
                         iNumLevel = 20, sColorMap = "RdYlBu_r" )
    
    return

# *********************** Zusammenstellen der Daten für Mittelwert/Median über eine Zeitspanne und einer eventuellen ***********************
###                            Eingrenzung der Sensoren anhand der Lage in einem umschreibenden Rechteck
def AnalyzeTimeAverageDataCV ( sDate, sSubFolderMonthYear, sDataSelection, tNumLags, tMaxLags, tModels, sAverageParameter = "median", 
                               tBorderX = None, tBorderY = None ):
    tParameterEstimator = ( ( "cressie", "scott", 16000 ), ( "matheron", "scott", 16000 ) )#, ( "dowd", "scott", 16000 ) )
    if ( tNumLags is None ):
        tNumLags = ( 5, 15 )
    else:
        CheckAssert ( bBool = ( len ( set ( tNumLags ) ) == 2 ), sMsg = "Invalid Shape <tNumLags>!" )
    if ( tMaxLags is None ):
        tMaxLags = ( 6000, 15000, 50 )
    else:
        CheckAssert ( bBool = ( len ( set ( tMaxLags ) ) == 3 ) , sMsg = "Invalid Shape <tMaxLags>!" )
        
    if ( tModels is None ):
        tModels = ( "spherical", "matern", "stable" )

    aSelectIndices, tLabel, sDataType = ga.GDictDataColumns.get ( sDataSelection )
    
    aRawData, DictStatistic = ds.LoadDataStatisticMonth ( sDataType = sDataType, sSubFolderMonthYear = sSubFolderMonthYear, sAverageParameter = sAverageParameter )
    aDataID = np.arange ( start = 1, stop = len ( DictStatistic ) + 1 )
    
    DictStatistic_inv = { tTupel[ 0 ][ 0 ]: sKey for sKey, tTupel in DictStatistic.items () }
    aData_sel = np.asarray ( aRawData[ :, aSelectIndices ], dtype = np.float64 )
    
    if ( ( tBorderX is not None ) and ( tBorderY is not None ) ):
        tStyleRectangle = ( tBorderX[ 0 ], tBorderY[ 0 ], tBorderX[ 1 ] - tBorderX[ 0 ], tBorderY[ 1 ] - tBorderY[ 0 ], 2.0, "-.", "r12", "none" )

        aSelectX = np.logical_and ( aData_sel[ :, 1 ] >= tBorderX[ 0 ], aData_sel[ :, 1 ] <= tBorderX[ 1 ] )
        aSelectY = np.logical_and ( aData_sel[ :, 0 ] >= tBorderY[ 0 ], aData_sel[ :, 0 ] <= tBorderY[ 1 ] )
        aSelect = np.logical_and ( aSelectX, aSelectY )
        print ( ">> Found Sensors: %d" % ( aSelect.shape ) )
        aRawData = aRawData[ aSelect, : ]
        aDataID = aDataID[ aSelect ]
        print ( ">> New Data Shape: %s" % ( str ( aRawData.shape ) ) )
        aData_sel = aRawData[ :, aSelectIndices ]
    else:
        tStyleRectangle = None
    
    fMinX, fMaxX = np.amin ( aData_sel[ :, 0 ] ), np.amax ( aData_sel[ :, 0 ] )
    fMinY, fMaxY = np.amin(  aData_sel[ :, 1 ] ), np.amax ( aData_sel[ :, 1 ] ) 
    
    #ga.ShowDataSnapshotwBorder ( aRawData = aRawData, sDataSelection = sDataSelection, sDate = sDate, CProjCRS = CCRS_azmequi, 
    #                             tStyleRectangle = tStyleRectangle, bShowDistribution = True )
    
    #ga.ShowDataSnapShotwGraph ( aRawData = aRawData, sDate = sDate, sDataSelection = sDataSelection, CProjCRS = CCRS_azmequi, bShowDistribution = True )
    
    print ( ">> Shape Data_sel: %s" % ( str ( aData_sel.shape ) ) )
    
    fMax, iArgMax = np.amax ( aData_sel[ :, 2 ] ), np.argmax ( aData_sel[ :, 2 ] )
    sSensorID_max = DictStatistic_inv.get ( aDataID[ iArgMax ] )
    print ( ">> ShowDataTimeAverage > Max: %.2f (ID: %s)" % ( fMax, sSensorID_max ) )
    #ShowDataSingleID ( sSensorID = sSensorID_max, tSequence = ( "2026-04-01", "2026-04-30" ), sDataSelection = sDataSelection )
    
    #ga.ShowDistributionDistances ( aData = aData_sel, sDescription = tLabel[ 2 ], sUnit = tLabel[ 3 ] )
    CVario = sa.CVariogramSkG ( aData = aData_sel, sEstimator = "cressie", bUseNugget = True, iNumLags = 9, sColorMap = None )

    #ListVariogramEstimation = sModel, sEstimator, iNumLags, uMaxLag, fRange, fSill, fNugget, fMSE
    ListVariogramEstimation = CVario.ScreenEstimationParameter ( tNumLags = tNumLags, tMaxLags = tMaxLags, tModels = tModels, 
                                                                 iShowEachFit = 500 )
    ### für Debugging genutzt
    #print ( len ( ListVariogramEstimation ) )
    #np.save ( file = "Data_sel.npy", arr = aData_sel )
    #np.save ( file = "ListVariogram.npy", arr = ListVariogramEstimation )
    
    # iNumLag, MaxLag, Range, Sill, Nugget, mean(MSE)
    tBestResults = RunIsotropicCrossValidation ( aData = aData_sel, ListVariogramEstimation = ListVariogramEstimation, 
                                        fLowerFitBoundRange = CVario.ListFitBounds[ 0 ][ 0 ], sDate = sDate, 
                                        sDescription = tLabel[ 2 ], bShowPlot = True )
    
    CVario.ShowVariogramEstimation ( tParameterEstimator = ( ( "cressie", int ( tBestResults[ 0 ] ), tBestResults[ 1 ] ) ) )      
    #CVario.ShowVariogramEstimation ( tParameterEstimator = ( ( "matheron", 9, 20000 ), ( "cressie", 9, 20000 ) ) )       

    

    ## aRank = fMSE, fVar, fLenScale, fNugget, fRescale, fShape
    aRank, ListRankModel = CVario.CompareCovModel4 ()
    print ( aRank )
    print ( ListRankModel )
    
    for ik in ( 0, 1, 2, 3 ):
        fShape = None
        sModel = ListRankModel[ ik ]
        fVar = aRank[ ik ][ 1 ]
        fLenScale = aRank[ ik ][ 2 ]
        fNugget = aRank[ ik ][ 3 ]
        fRescale = aRank[ ik ][ 4 ]
        if ( sModel in [ "stable", "matern" ] ):
            fShape = aRank[ ik ][ 5 ]

        sInfoBoxText = "CV MSE: %.2f" % ( aRank[ ik ][ 0 ] )
        CCovModel = sa.CCovarianceModelGsT ( sModel = sModel, fVar = fVar, uLenScale = fLenScale, fNugget = fNugget, 
                                             fRescale = fRescale, fShape = fShape )
        CKrige = sa.COrdKrigingGsT ( CCovarianceModel = CCovModel, uDataObserved = np.transpose ( aData_sel ), bFitVariogram = False )
        CInfobox = pl.CInfoBox ( sText = sInfoBoxText, fBoxPosX = -11800, fBoxPosY= -6200, iBoxFontSize = 12, sFaceColor = "chartreuse" )
        CKrige.Interpolate ( tDimX = ( fMinX, fMaxX, 400 ), tDimY = ( fMinY, fMaxY, 400 ), tLimX = ( 300, 300 ), tLimY = ( 300, 300 ), 
                             iNumLevel = 40, sColorMap = "RdYlBu_r", CInfoBox = CInfobox, bShowVariance = True )
    
    return
# ********************** Zusammenstellen der Daten für einen Zeitpunkt plus-minus delta Minuten und einer eventuellen **********************
###                            Eingrenzung der Sensoren anhand der Lage in einem umschreibenden Rechteck
def AnalyzeTimeFrameData ( sDate, sStartTime, sDataSelection, tParameterEstimator = None, tBorderX = None, tBorderY = None, iWindowDelta = 2, 
                           bUseMeanMatches = False ):
    if ( tParameterEstimator is None ):
        tParameterEstimator = ( ( "ward", 10000 ), ( "fd", 12000 ), ( "uniform", 16000 ) )
    else :
        CheckAssert ( bBool = ( len ( tParameterEstimator ) == 3 ), sMsg = "Invalid Shape <tParameterEstimator>! " ) 
        
    sBinFunc1, tMaxLags = tParameterEstimator[ 0 ][ 0 ], tParameterEstimator[ 0 ][ 1 ]
    uR_max11, uR_max12, uR_max13 = tMaxLags
    sBinFunc2, tMaxLags = tParameterEstimator[ 1 ][ 0 ], tParameterEstimator[ 1 ][ 1 ]
    uR_max21, uR_max22, uR_max23 = tMaxLags
    sBinFunc3, tMaxLags = tParameterEstimator[ 2 ][ 0 ], tParameterEstimator[ 2 ][ 1 ]
    uR_max31, uR_max32, uR_max33 = tMaxLags
    aSelectIndices, tLabel, sDataType = ga.GDictDataColumns.get ( sDataSelection )
    aRawData, aDataID, DictInfo = ds.LoadDataTimeFrame ( sDate = sDate, sStartTime = sStartTime, sDataType = sDataType, iWindowDelta = iWindowDelta, 
                                                         bUseMeanMatches = bUseMeanMatches )
    
    DictInfo_inv = { tTupel[ 0 ]: sKey for sKey, tTupel in DictInfo.items () }
    aData_sel = aRawData[ :, aSelectIndices ]
    
    if ( tBorderX is not None and tBorderY is not None ):
        tStyleRectangle = ( tBorderX[ 0 ], tBorderY[ 0 ], tBorderX[ 1 ] - tBorderX[ 0 ], tBorderY[ 1 ] - tBorderY[ 0 ], 2.0, "-.", "r12", "none" )

        aSelectX = np.logical_and ( aData_sel[ :, 1 ] >= tBorderX[ 0 ], aData_sel[ :, 1 ] <= tBorderX[ 1 ] )
        aSelectY = np.logical_and ( aData_sel[ :, 0 ] >= tBorderY[ 0 ], aData_sel[ :, 0 ] <= tBorderY[ 1 ] )
        aSelect = np.logical_and ( aSelectX, aSelectY )
        print ( ">> Found Sensors: %d" % ( aSelect.shape ) )
        aRawData = aRawData[ aSelect, : ]
        aDataID = aDataID[ aSelect ]
        print ( ">> New Data Shape: %s" % ( str ( aRawData.shape ) ) )
        aData_sel = aRawData[ :, aSelectIndices ]
    else:
        tStyleRectangle = None
    
    fMinX, fMaxX = np.amin ( aData_sel[ :, 0 ] ), np.amax ( aData_sel[ :, 0 ] )
    fMinY, fMaxY = np.amin(  aData_sel[ :, 1 ] ), np.amax ( aData_sel[ :, 1 ] ) 
    
    ga.ShowDataSnapshotwBorder ( aRawData = aRawData, sDate = sDate, sStartTime = sStartTime, sDataSelection = sDataSelection, 
                                 iWindowDelta = iWindowDelta, CProjCRS = CCRS_azmequi, tStyleRectangle = tStyleRectangle, 
                                 bShowDistribution = False )
    
    print ( ">> Shape Data_sel: %s" % ( str ( aData_sel.shape ) ) )
    
    fMax, iArgMax = np.amax ( aData_sel[ :, 2 ] ), np.argmax ( aData_sel[ :, 2 ] )
    sSensorID_max = DictInfo_inv.get ( aDataID[ iArgMax ] )
    print ( ">> ShowDataTimeFrame > Max: %.2f (ID: %s)" % ( fMax, sSensorID_max ) )
    ShowDataSingleID ( sSensorID = sSensorID_max, sDate = sDate, sDataSelection = sDataSelection )
    
    ga.ShowDistributionDistances ( aData = aData_sel, sDescription = tLabel[ 2 ], sUnit = tLabel[ 3 ] )
    
    
    CVario = sa.CVariogramSkG ( aData = aData_sel, sEstimator = "cressie", bUseNugget = True, iNumLags = 15, sColorMap = None )
    CVario.ShowVariogramEstimation ( tParameterEstimator = ( ( "cressie", sBinFunc1, uR_max11 ), 
                                                             ( "cressie", sBinFunc1, uR_max12 ), 
                                                             ( "cressie", sBinFunc1, uR_max13 ) ) )
    CVario.ShowVariogramEstimation ( tParameterEstimator = ( ( "cressie", sBinFunc2, uR_max21 ), 
                                                             ( "cressie", sBinFunc2, uR_max22 ), 
                                                             ( "cressie", sBinFunc2, uR_max23 ) ) )
    CVario.ShowVariogramEstimation ( tParameterEstimator = ( ( "cressie", sBinFunc3, uR_max31 ), 
                                                             ( "cressie", sBinFunc2, uR_max32 ), 
                                                             ( "cressie", sBinFunc3, uR_max33 ) ) )

    aRank, ListRankModel = CVario.CompareCovModel ()
    
    CVario.CheckIsotropy ( sEstimator = "cressie", sModel = ListRankModel[ 0 ], iNumSamplePoints = None, uMaxLag = uR_max32, iNumLags = 15, 
                           sBinFunc = "ward" )
    
    print ( ListRankModel[ 0 ] )
    CCovModel = sa.CCovarianceModelGsT ( sModel = ListRankModel[ 0 ], fVar = 10.0, uLenScale = 100.0, fNu = 1.5, fAlpha = 1.0, fNugget = 2.1 )
    CKrige = sa.COrdKrigingGsT ( CCovarianceModel = CCovModel, uDataObserved = np.transpose ( aData_sel ), bFitVariogram = True )
    CKrige.Interpolate ( tDimX = ( fMinX, fMaxX, 400 ), tDimY = ( fMinY, fMaxY, 400 ), tLimX = ( 300, 300 ), tLimY = ( 300, 300 ),
                         iNumLevel = 20, sColorMap = "RdYlBu_r" )
    
    return
# ************************************** Verlauf der Stundenmittelwerte der LUA Sensoren im April 2026 *************************************
def ShowDataSensorLUA ( sMonthYear = "Apr2026" ):
    aDateTime, aData, DictInfo = ds.ImportSensorDataMatterLUA ( sMonthYear = sMonthYear )
    
    tStyleP10 = ( "c12", "o", 4.0, "--", 1.0, "$\mathrm{PM}_{10}$" )
    tStyleP2x5 = ( "b12", "D", 4.0, "--", 1.0, "$\mathrm{PM}_{2.5}$" )
    CHLine1 = pl.CLine ( sLineColor = "r12", fLinePos = 25.0, fLineWidth = 2.0, sLineStyle = "--", sLineLabel = "Jahresgrenzwert $\mathrm{PM}_{2.5}$" )
    GraCon = pl.CGraphicConfig ( sTitle = "Feinstaubmessungen Landesumweltamt April 2026: München Stacchus", sLabelX = "Zeit", 
                                 sLabelY = "Stundenmittelwerte $\mathrm{PM}_{10}$ und $\mathrm{PM}_{2.5}$", HLine1 = CHLine1 )

    pl.PlotX2Y ( aX = aDateTime, aY1 = aData[ :, 6 ], aY2 = aData[ :, 7 ], tStyleY1 = tStyleP2x5, tStyleY2 = tStyleP10, GraphicConfig = GraCon )
    

    ### Stundenmittelwerte PM2.5
    sLabelYPM2x5 = "Stundenmittelwerte $\mathrm{PM}_{2.5}$"
    GraConQuartett = pl.CGraphicConfig ( sLabelY = sLabelYPM2x5, sLabelY3 = sLabelYPM2x5 )#, sLabelY3 = sLabelYPM2x5, sLabelY4 = sLabelYPM2x5 )

    fMeanJoh2x5 = np.mean ( aData[ :, 0 ] )    
    fMeanLan2x5 = np.mean ( aData[ :, 2 ] )    
    fMeanLot2x5 = np.mean ( aData[ :, 4 ] )
    fMeanSta2x5 = np.mean ( aData[ :, 6 ] )
    CHLineJoh = pl.CLine ( sLineColor = "r12", fLinePos = fMeanJoh2x5, sLineStyle = "--", fLineWidth = 2.0, sLineLabel = "Monats-Mittelwert" )
    CHLineLan = pl.CLine ( sLineColor = "r12", fLinePos = fMeanLan2x5, sLineStyle = "--", fLineWidth = 2.0, sLineLabel = "Monats-Mittelwert" )
    CHLineLot = pl.CLine ( sLineColor = "r12", fLinePos = fMeanLot2x5, sLineStyle = "--", fLineWidth = 2.0, sLineLabel = "Monats-Mittelwert" )
    CHLineSta = pl.CLine ( sLineColor = "r12", fLinePos = fMeanSta2x5, sLineStyle = "--", fLineWidth = 2.0, sLineLabel = "Monats-Mittelwert" )
    GraConQuartett.Set ( HLine1 = CHLineJoh, HLine2 = CHLineLan, HLine3 = CHLineLot, HLine4 = CHLineSta )

    pl.PlotQuartett ( uX = aDateTime, tY = ( aData[ :, 0 ], aData[ :, 2 ], aData[ :, 4 ], aData[ :, 6 ] ), GraphicConfig = GraConQuartett, 
                      tStyles = ( ( "b12", "o", 4.0, "-", 1.0, "Johanneskirchen" ),
                                  ( "b12", "o", 4.0, "-", 1.0, "Landshuter Allee" ),
                                  ( "b12", "o", 4.0, "-", 1.0, "Lothstraße" ),
                                  ( "b12", "o", 4.0, "-", 1.0, "Stacchus" ) ) )
    
    GraCon = pl.CGraphicConfig ( sTitle = "Feinstaubmessungen Landesumweltamt in München: April 2026", sLabelX = "Zeit",
                                 sLabelY = "Stundenmittelwerte $\mathrm{PM}_{2.5}$" )
    
    pl.PlotX4Y ( aX = aDateTime, tY = ( aData[ :, 0 ], aData[ :, 2 ], aData[ :, 4 ], aData[ :, 6 ] ), 
                 tStylesY = ( ( "b12", "o", 1.0, "-", 1.0, "Johanneskirchen" ),
                            ( "o12", "D", 1.0, "-", 1.0, "Landshuter Allee" ),
                            ( "c12", "o", 1.0, "-", 1.0, "Lothstraße" ),
                            ( "s12", "D", 1.0, "-", 1.0, "Stacchus" ) ), GraphicConfig = GraCon )

    return
# ***************************************** Grafische Darstellung des Verlaufes für einen Sensor *******************************************
def ShowDataSingleID ( sSensorID, sDataSelection, tSequence = None, sDate = None, sSubFolderMonthYear = "Apr2026" ):
    tStyleY1 = ( "o12", "o", 4.0, "", 0.0, "" )
    tStyleY2 = ( "c12", "o", 4.0, "", 0.0, "" )
    tSelectIndices, tLabel, sDataType = ga.GDictDataColumns.get ( sDataSelection )
    
    aDateTime, aData, tInfo = ds.LoadData ( sSensorID = sSensorID, tSequence = tSequence, sDate = sDate, sDataType = sDataType,
                                            sSubFolderMonthYear = sSubFolderMonthYear )

    plt.plot ( aDateTime, aData[ :, 0 ] )
    plt.show ()
    
    iCountID, fLon, fLat, fLon_ref, fLat_ref = tInfo 
        
    sTextLabelY = tLabel[ 2 ] + " (" + tLabel[ 3 ] + ")"
    #sTitleText = "Sensor-ID: %s (%d); Pos: Geo: (%.2f, %.2f), Azimutal: (%.2f, %.2f)" % ( sSensorID, iCountID, fLat, fLon, fLon_ref, fLat_ref ) 
    sTitleText = "Sensor-ID: %s (%d); Azimutal: (%.2f, %.2f)" % ( sSensorID, iCountID, fLon_ref, fLat_ref ) 
    
    if ( len ( tSelectIndices )  == 3 ): ## einzelner Messwert für P1, P2 oder T
        CGraCon = pl.CGraphicConfig ( sTitle = sTitleText, sLabelX = "Zeit", sLabelY = sTextLabelY )
        fMean = np.mean ( aData[ :, 0 ] )
        fMedian = np.median ( aData[ :, 0 ] )
        CHLine1 = pl.CLine ( sLineColor = "b12", fLinePos = fMean, sLineLabel = "Mittelwert", fLineWidth = 2.0, sLineStyle = "--" )
        CHLine2 = pl.CLine ( sLineColor = "p8", fLinePos = fMedian, sLineLabel = "Median", fLineWidth = 2.0, sLineStyle = "-" )
        CGraCon.Set ( HLine1 = CHLine1, HLine2 = CHLine2 )
        pl.PlotXY ( aX = aDateTime, aY = aData[ :, 0 ], tStyle = tStyleY1, GraphicConfig = CGraCon ) 
    else: ## beide Messerte P1 und P2
        sTextLabelY2 = tLabel[ 4 ] + " (" + tLabel[ 5 ] + ")"
        CGraCon = pl.CGraphicConfig ( sTitle = sTitleText, sLabelX2 = "Zeit", 
                                      sLabelY = sTextLabelY, sLabelY2 = sTextLabelY2 )
        pl.PlotStackVertX2Y ( aX = aDateTime, aY1 = aData[ :, 0 ], aY2 = aData[ :, 1 ], GraphicConfig = CGraCon, 
                              tStyle1 = tStyleY1, tStyle2 = tStyleY2 )
    
    if ( sDate is not None ):
        sText = "Sensor-ID %s (%s)" % ( sSensorID, sDate )
    else:
        sText = "Sensor-ID %s" % ( sSensorID )
    ga.ShowDistributionMeasurements ( aData = aData[ :, 0 ], sTitleStartText = sText, sDescription = tLabel[ 2 ], 
                                      sUnit = tLabel[ 3 ], iNumBins = 40 )
    
    return ( aDateTime, aData )

def ShowDataSingleID2 ( sSensorID, sDataSelection, aDateTimeAll, aDataAll, aID_All, DictInfo ):
    tStyleY1 = ( "o12", "o", 4.0, "", 0.0, "" )
    tStyleY2 = ( "c12", "o", 4.0, "", 0.0, "" )
    tSelectIndices, tLabel, sDataType = ga.GDictDataColumns.get ( sDataSelection )
    
    tInfo = DictInfo.get ( sSensorID )
    iCountID, fLon, fLat, fLon_ref, fLat_ref, tAvailableDays = tInfo 
    
    aSelectID = ( aID_All == iCountID )
    aData = aDataAll[ aSelectID ]
    aDateTime = aDateTimeAll[ aSelectID ]
            
    sTextLabelY = tLabel[ 2 ] + " (" + tLabel[ 3 ] + ")"
    sTitleText = "Sensor-ID: %s (%d); Pos: Geo: (%.2f, %.2f), Azimutal: (%.2f, %.2f)" % ( sSensorID, iCountID, fLat, fLon, fLon_ref, fLat_ref ) 
    
    if ( len ( tSelectIndices )  == 3 ): ## einzelner Messwert für P1, P2 oder T
        CGraCon = pl.CGraphicConfig ( sTitle = sTitleText, sLabelX = "Zeit", sLabelY = sTextLabelY )
        pl.PlotXY ( aX = aDateTime, aY = aData[ :, 0 ], tStyle = tStyleY1, GraphicConfig = CGraCon ) 
    else: ## beide Messerte P1 und P2
        sTextLabelY2 = tLabel[ 4 ] + " (" + tLabel[ 5 ] + ")"
        CGraCon = pl.CGraphicConfig ( sTitle = sTitleText, sLabelX2 = "Zeit", 
                                      sLabelY = sTextLabelY, sLabelY2 = sTextLabelY2 )
        pl.PlotStackVertX2Y ( aX = aDateTime, aY1 = aData[ :, 0 ], aY2 = aData[ :, 1 ], GraphicConfig = CGraCon, 
                              tStyle1 = tStyleY1, tStyle2 = tStyleY2 )
    
    return ( aDateTime, aData )
# ********************* Analyse der Zeitreihen auf Auffälligkeiten hin und ggf Imputation der Artefakte und Ausreißer **********************
def ScreenData ( sDate, sDataSelection, sSubFolderMonthYear, tSetSensorID = None, tDegreeRollMean = ( 10, 10 ), fMaxInterruption = 60.0, 
                 tTolerance = ( 2.0, 0.05 ), ### Toleranzfaktor und Anteil der akzeptierten Ausreißer
                 tPlateauDetection = ( 50.0, 4 ), 
                 bShow = True, bShowSummaryPlot = True ):
    fThreshold, iMaxConsecutiveValuesAboveThreshold = tPlateauDetection
    fToleranceFactor, fAcceptedOutlierRatio = tTolerance
    
    aSelectIndices, tDescription, sDataType = ga.GDictDataColumns.get ( sDataSelection )
    sMesswert, sUnit = tDescription[ 2 ], tDescription[ 3 ]
    aDateTimeAll, aDataAll, aID_All, DictInfo, DictDays = ds.LoadRawData ( sDate = sDate, sDataType = sDataType, sSubFolderMonthYear = sSubFolderMonthYear )
    
    if ( tSetSensorID is None ):
        tSetSensorID = DictInfo.keys ()
        
    if ( aDataAll.shape[ 1 ] == 2 ): ### Feinstaub Daten
        iDataIndex = int ( sDataSelection[ -1 ] ) - 1 ## holt den letzten Buchstaben aus dem String: also 1 für AzmP1, bzw. 2 für AzmP2
    else:
        iDataIndex = 0

    ListResult = list ()
    
    for sSensorID in tSetSensorID:
        iCountID = DictInfo.get ( sSensorID ) [ 0 ]
        aSelectID = ( aID_All == iCountID )
        aRawData = aDataAll[ aSelectID ]
        aDateTime = aDateTimeAll[ aSelectID ]
        sTag = None
        ### Wichtig: hier werde die beiden Datein mit den nan Werten korrigiert!
        #aDateTime, aRawData = ShowDataSingleID ( sSensorID = sSensorID, sDataSelection = sDataSelection, sDate = sDate )

        aData = np.asarray ( aRawData[ :, iDataIndex ], dtype = np.float64 )
        #print ( ">> Shape Data: %s" % ( str ( aData.shape ) ) )
        
        iNumAcceptedOutlier = int ( float ( aData.shape[ 0 ] ) * fAcceptedOutlierRatio )
        fStd, fMax = np.nanstd ( aData ), np.nanmax ( aData )
        aRollMean = CalcRollingMean ( aData = aData, tDegree = tDegreeRollMean, sPadMode = "median", uStatLength = ( 5, 5 ) )
        
        sTitleTextStart = "Sensor-ID %s [%d]: " % ( sSensorID, iCountID )
        #sTitleTextStart = "Sensor-ID %s (%s): " % ( DictID.get ( iCountID )[ 0 ], sDate )
        ListMarker = list ()
        aData_imp = np.copy ( aData )
        
        DictResult = ScreenDataSeriesAbnormality ( aDateTime = aDateTime, aData = aData, aReferenceData = aRollMean, 
                                                      fToleranceValue = fToleranceFactor * fStd, iOutlierNumNeighbours = 0, fThreshold = fThreshold )
        
        iNumNA = DictResult.get ( "NumberNA" )
        iNumDifferentValues = DictResult.get ( "NumberDifferentValues" )
        fMaxInterruptionMin = ( DictResult.get ( "TimeDeltaQuartiles" )[ 4 ] / 60.0 )
        aSequenceLengthThreshold = DictResult.get ( "SequenceLengthThreshold" )
        aIndicesOutlier = DictResult.get ( "IndicesOutlier" )
        aIndicesThreshold = DictResult[ "IndicesThreshold" ] 
        
        ### Schritt 1: Anzahl der np.nan Werte
        if ( ( iNumNA > 0 ) and ( sTag is None ) ):
            sTag = "%d NA Werte $\\rightsquigarrow$ verworfen" % ( iNumNA )
            sCategoryLabel = "verworfen:\nmehrere\nNA-Werte"

        ### Schritt 2: Anzahl der verschiedenen Werte 
        if ( ( iNumDifferentValues < 20 ) and ( sTag is None ) ):
            sTag = "nur %d verschiedene Werte $\\rightsquigarrow$ verworfen" % ( iNumDifferentValues )
            sCategoryLabel = "verworfen:\nzu wenige\nvers. Werte"
        
        ### Schritt 3: Unterbrechungen von mehr als fMaxInterruption Minuten werden verworfen
        if ( ( fMaxInterruptionMin > fMaxInterruption ) and ( sTag is None ) ):
            sTag = "%.0f Minuten Unterbrechung $\\rightsquigarrow$ prüfen" % ( fMaxInterruptionMin )
            sCategoryLabel = "verworfen:\nzeitliche\nLücken"
        
        ### Schritt 4: ZUSAMMENHÄNGENDE Plateaus von Werte OBERHALB eines Schwellwerts werden verworfen
        if ( aSequenceLengthThreshold is not None ):
            iMaxSequenceLengthThreshold = aSequenceLengthThreshold[ 0 ] + 1
            if ( ( iMaxSequenceLengthThreshold > iMaxConsecutiveValuesAboveThreshold ) and ( sTag is None ) ):
                sTag = "Sequenz Länge %d über Schwellwert $\\rightsquigarrow$ prüfen" % ( ( iMaxSequenceLengthThreshold ) )
                sCategoryLabel = "verworfen:\noberhalb\nSchwellwert"
                print ( ">> Sequence-Length-Check > CountID: %d, LengthThreshold: %s" % ( iCountID, str ( aSequenceLengthThreshold ) ) )
                aIndicesOutlier = aIndicesThreshold
        
        ### Schritt 5: Bestimmung der Ausreißer anhand des gleitenden Durchschnitts falls es mehr sind als fAcceptedRatio Prozent
        if ( sTag is None ):
            if ( aIndicesOutlier.shape[ 0 ] > iNumAcceptedOutlier ):
                sTag = "vorgesehen für Bereinigung"
            else:
                sTag = "ok"
                sCategoryLabel = "verwendet:\nqualitativ ok"
                ListMarker = None

        ### 4. Schritt: Entfernung der erweiterten Ausreißer und Imputation der nun fehlenden Ausreißer durch kubische Splines 
        if ( sTag == "vorgesehen für Bereinigung" ):
            aX = np.arange ( start = 0, stop = aData.shape[ 0 ] )
            aX_use = np.setdiff1d ( ar1 = aX, ar2 = aIndicesOutlier, assume_unique = True )
            aData[ aX_use ] = aRollMean[ aX_use ] ### klappt besser als die Spline-Interpolation direk auf den Daten
            CPPoly = CubicSpline ( aX_use, aData[ aX_use ] )
            aData_cs = CPPoly ( aX )
            aData_imp[ aIndicesOutlier ] = aData_cs[ aIndicesOutlier ]
            for iIndexOutlier in aIndicesOutlier:
                ListMarker.append ( ( "X", aDateTime[ iIndexOutlier ], aData_imp[ iIndexOutlier ], 10.0, "o10", "o10" ) )
                     
            sTag = "bereinigt"
            sCategoryLabel = "verwendet:\nbereinigt"
              
        ListResult.append ( ( sSensorID, iCountID, sCategoryLabel ) )
        
        if ( bShow == True ):
            if ( fMax > fThreshold ):
                CHLine = pl.CLine ( sLineColor = "p10", fLinePos = fThreshold, sLineLabel = "Schwellwert %.0f %s" % ( fThreshold, sUnit ) )
            else:
                CHLine = None
            
            CGraCon = pl.CGraphicConfig ( sLabelX = "Zeit", sLabelY = sMesswert + " (" + sUnit + ")", HLine1 = CHLine )
            tFillArea = ( aRollMean + fToleranceFactor * fStd, aRollMean - fToleranceFactor * fStd, "g10", 0.2, 
                          "Toleranzband ($\lambda_{\\text{tol}}=%.0f$)" % ( fToleranceFactor ) )
        
            if ( sTag == "bereinigt" ):
                sTitleText = sTitleTextStart + "%d Ausreißer " % ( aIndicesOutlier.shape[ 0 ] ) + "$\\rightsquigarrow$ " + sTag
                sLabelY1 = "bereinigte Daten"
                CGraCon.Set ( sTitle = sTitleText, sMarkerSingleLabel = "bereinigter Ausreißer" )
                pl.Plot2X3Y ( aX1 = aDateTime, aX2 = aDateTime[ aIndicesOutlier ], aY1 = aData_imp, aY2 = aRollMean, aY3 = aData[ aIndicesOutlier ],
                              tStyleY1 = ( "c10", "o", 6.0, "", 0.0, sLabelY1 ), 
                              tStyleY2 = ( "b12", "o", 2.0, "-.", 2.0, "gleitender Durchschnitt (m=%d)" % ( tDegreeRollMean[ 0 ] ) ), 
                              tStyleY3 = ( "p13", "o", 7.0, "", 0.0, "Ausreißer" ), 
                              GraphicConfig = CGraCon, 
                              tFillArea = tFillArea, ListMarker = ListMarker, ListAnnotation = None )
            else:
                sTitleText = sTitleTextStart + sTag
                CGraCon.Set ( sTitle = sTitleText )
                pl.PlotX2Y ( aX = aDateTime, aY1 = aData_imp, aY2 = aRollMean, 
                             tStyleY1 = ( "c10", "o", 6.0, "", 0.0, "original Daten" ), 
                             tStyleY2 = ( "b12", "o", 0.0, "-.", 3.0, "gleitender Durchschnitt (m=%d)" % ( tDegreeRollMean[ 0 ] ) ), 
                             GraphicConfig = CGraCon, tFillArea = tFillArea, ListMarker = None, ListAnnotation = None )
    
    DictCategoryLabel = dict ()
    for ik in range ( len ( ListResult ) ):
        tTupel = ListResult[ ik ]
        sLabel = tTupel[ 2 ]
        if ( sLabel in DictCategoryLabel.keys () ):
            DictCategoryLabel[ sLabel ] = DictCategoryLabel[ sLabel ] + 1
        else:
            DictCategoryLabel[ sLabel ] = 1
        
    DictCategoryLabel = SortDict ( DictCategoryLabel )
    
    if ( bShowSummaryPlot == True ):
        ListAnnotationSummary = list ()
        
        CGraCon = pl.CGraphicConfig ( sTitle = "Analyse der %s Messungen vom %s (N=%d)" % ( sMesswert, sDate, len ( tSetSensorID ) ), 
                                     sLabelY = "Anzahl", sGridAxis = "y", sAnnotationHorzAlign = "center", 
                                     sAnnotationVertAlign = "bottom" )
        aX = np.asarray ( list ( DictCategoryLabel.keys () ) )
        aY = np.asarray ( list ( DictCategoryLabel.values ( ) ) )
        
        for ik in range ( aX.shape[ 0 ] ):
            ListAnnotationSummary.append ( ( str ( aY[ ik ] ), aX[ ik ], aY[ ik ], 16.0, "black" ) )

        #pl.PlotStackedBarChart ( aX = aX, aData = aY, GraphicConfig = CGraCon, sColor = "b11" )
        pl.PlotBarChart ( aX = aX, aData = aY, GraphicConfig = CGraCon, ListAnnotation = ListAnnotationSummary, uColor = "b11" )
        
    return ( ListResult )
# ********* Durchführung einer Cross-Validation zur Bestimmung der besten Parameter-Kombination für die isotrope Variogramm-Schätzung ******
def RunIsotropicCrossValidation ( aData, ListVariogramEstimation, fLowerFitBoundRange, sDate, sDescription, bShowPlot = True ):
    CheckAssert ( bBool = ( len ( ListVariogramEstimation ) > 0 ), sMsg = "<ListVariogramEstimation> is empty!" )
    ListResultCV = list ()
    fTolFitBound = 0.01
    iInvalidCounter = 0
    
    for ik, tTupel in enumerate ( ListVariogramEstimation ):
        sModel, sEstimator, iNumLag, fMaxLag = tTupel[ 0 ], tTupel[ 1 ], tTupel[ 2 ], float ( tTupel[ 3 ] )
        fRange, fSill, fNugget, fShape =  float ( tTupel[ 4 ] ), float ( tTupel[ 5 ] ), float ( tTupel[ 6 ] ), float ( tTupel[ 7 ] )
        ## wenn Range nahe bei maxlag liegt, wurden die Boundaries gefittet; analog wenn in der Nähe des lower bounds
        if ( ( fRange < ( 1.0 + fTolFitBound ) * fLowerFitBoundRange ) ):
        #if ( fRange > ( 1.0 - fTolFitBound ) * fMaxLag ):  
        #if ( ( fRange > ( 1.0 - fTolFitBound ) * fMaxLag ) or ( fRange < ( 1.0 + fTolFitBound ) * fLowerFitBoundRange ) ):    
            iInvalidCounter += 1
            continue
        fShape = max ( fShape, 0.2 )
        CCovModelGSt = sa.CCovarianceModelGsT ( sModel = sModel, fVar = fSill, uLenScale = fRange, fNugget = fNugget, 
                                                fShape = fShape, bLatLon = False, bInfo = False  )

        CKrige = sa.COrdKrigingGsT ( CCovarianceModel_gst = CCovModelGSt.CCovModel_gst, uDataObserved = np.transpose ( aData ), 
                                     bFitVariogram = False )
        aMSE = CKrige.RunCrossValidation ( iNumFolds = 3, bShowInfo = False )
        ListResultCV.append ( ( sModel, sEstimator, iNumLag, fMaxLag, fRange, fSill, fNugget, np.mean ( aMSE ) ) )
    
    print ( ">> RunCrossValidation > %d of %d Fits were invalid!" % ( iInvalidCounter, len ( ListVariogramEstimation ) ) )
    
    aResultCV = np.asarray ( ListResultCV )
    aResultCV_str = aResultCV[ :, : 2 ]
    aResultCV_float = np.asarray ( aResultCV[ :, 2 : ], dtype = np.float64 )
    aResultCV_MSE = aResultCV_float[ :, -1 ] 
    
    aModel_used = np.unique ( aResultCV_str[ :, 0 ] )
    aEstimator_used = np.unique ( aResultCV_str[ :, 1 ] )    
    ListY = list ()
    ListX = list ()
    for sModel in aModel_used:
        for sEstimator in aEstimator_used:
            aSelect = np.logical_and ( aResultCV_str[ :, 0 ] == sModel, aResultCV_str[ :, 1 ] == sEstimator )
            aResultCV_MSE_sel = aResultCV_MSE[ aSelect ]
          
            ListY.append ( "(%s, %s)" % ( sModel.capitalize (), sEstimator ) )
            ListX.append ( aResultCV_MSE_sel )
        
    PlotResultCV ( ListX = ListX, ListY = ListY, fTotalMin = np.amin ( aResultCV_MSE ) )
    
    #print ( ListResultCV )
    # iNumLag, MaxLag, Range, Sill, Nugget, mean(MSE)
    aResultCV = np.asarray ( np.asarray ( ListResultCV )[ :, 2 : ], dtype = np.float32 )
    iArgMinTotal = np.argmin (  aResultCV[ :, -1 ] )
    
    print ( ListResultCV[ iArgMinTotal ] )
    print ( aResultCV[ iArgMinTotal ] )

    if ( bShowPlot == True ):
        sTitleText = "Cross-Validation Resultat (%s: %s Messungen)" % ( sDate, sDescription )
        GraCon = pl.CGraphicConfig ( sTitle = sTitleText, sLabelX = "Range", sLabelY = "Sill", 
                                     sLegend = "mittlerer MSE der CV" )
        pl.PlotScatterXY ( aX = aResultCV[ :, 2 ], aY = aResultCV[ :, 3 ], aZ = aResultCV[ :, 5 ], GraphicConfig = GraCon, 
                           tStyle = ( "RdYlBu_r", "o", 50.0, "" ) )
    
        GraCon.Set ( sLabelX = "$r_{max}$", sLabelY = "#Lags" )
        pl.PlotScatterXY ( aX = aResultCV[ :, 1 ], aY = aResultCV[ :, 0 ], aZ = aResultCV[ :, 5 ], sEdgeColor = "black",
                          GraphicConfig = GraCon, tStyle = ( "RdYlBu_r", "o", 50.0, "" ) )
    
    return ( aResultCV[ iArgMinTotal ] )

# **************** Eigener Plot zur Veranschaulichung der kategorialen CV Resultate für die Kombianation Schätzer / Modell *****************
def PlotResultCV ( ListX, ListY, fTotalMin ):
    
    print ( len ( ListX ) )
    
    tStyles = ( ( "s6", "o" ), ( "b6", "o" ), ( "s10", "o" ) , ( "b10", "o" ), ( "s14", "o" ), ( "b14", "o" ), ( "s18", "o" ), ( "b18", "o" ) )
    for aX, aY, tStyle in zip ( ListX, ListY, tStyles ):
        plt.plot ( aX, [ aY ] * len ( aX ), color = pl.GetColor ( tStyle[ 0 ] ), marker = tStyle[ 1 ], markersize = 9.0,
                  linestyle = "none" )
    plt.axvline ( x = fTotalMin, color = pl.GetColor ( "r12" ), ls = "--", lw = 2.0, label = "Kleinster MSE" )
    plt.title ( "Cross-Validation Resultat: Kombination (Kovarianz-Modell, Schätzer)", 
                fontname = pl.GDictPlotParameter.get ( "FontName" ), fontsize = pl.GDictPlotParameter.get ( "TitleSize" ) )
    plt.xlabel ( "MSE", fontname = pl.GDictPlotParameter.get ( "FontName" ), fontsize = pl.GDictPlotParameter.get ( "LabelSize" ) )
    plt.xticks ( fontname = pl.GDictPlotParameter.get ( "FontName" ), fontsize = pl.GDictPlotParameter.get ( "TickSize" )  )
    plt.yticks ( fontname = pl.GDictPlotParameter.get ( "FontName" ), fontsize = pl.GDictPlotParameter.get ( "TickSize" ) )
    plt.grid ( visible = True, axis = "both" )

    plt.legend ( prop = { "family": pl.GDictPlotParameter.get ( "FontName" ), "size": pl.GDictPlotParameter.get ( "LegendSize" ) }, 
             loc = "best" )
    plt.show ()

    return
# ******************************* Auffinden der nächstgelegenen Sensoren bezogen auf einen Ausgangspunkt *********************************** 
def FindNearestLocation ( sRefKey ):
    DictLoc = ds.LoadAllSensorLocations ()
    
    ListPoints = list ()
    ListDist = list ()
    ListKeys = list ()
    
    fLat_ref_LUA, fLon_ref_LUA = DictLoc.get ( sRefKey )[ 3 : ]

    aPoint_LUA = np.asarray ( [ fLat_ref_LUA, fLon_ref_LUA ] )

    for sKey in DictLoc.keys ():
        if ( sKey == sRefKey ):
            continue
    
        tInfo = DictLoc.get ( sKey )
        fLat_ref, fLon_ref = tInfo[ 3 ], tInfo[ 4 ]
        aPoint = np.asarray ( [ fLat_ref, fLon_ref ] )
    
        fDistance = np.linalg.norm ( aPoint_LUA - aPoint )
        ListPoints.append ( aPoint )
        ListKeys .append ( sKey )
        ListDist.append ( fDistance ) 
    
    aDistance = np.asarray ( ListDist, dtype = np.float64 )
    aPoints = np.asarray ( ListPoints, dtype = np.float64 )
    aKeys = np.asarray ( ListKeys, dtype = str )
    aIndices = np.argsort ( a = aDistance )

    return ( aPoints[ aIndices ], aDistance[ aIndices ], aKeys[ aIndices ] )
# ****************** Grafische Aufbereitung der Stunden-Mittelwerte einer LUA Station und der umgebenden SDS011 Sensoren ******************* 
def ShowSensorNearLUA ( sKeyRefLUA ):
    CheckAssert ( bBool = ( sKeyRefLUA in [ "Joh", "Lan" ] ), sMsg = "Invalid Choice <sKeyRefLUA>!" )
    DictLoc = { "Joh" : "Johanneskirchen", "Lan" : "Landshuter Allee" }
    sMonthYear = "Apr2026"
    
    ListDataAverage = list ()
    ListLabels = list ()
    
    _, aDist, aKeys = FindNearestLocation ( sRefKey = sKeyRefLUA )
    aDistTop3 = aDist[ : 3 ]
    aKeysTop3 = aKeys[ : 3 ] 
    for fDist in aDistTop3:
        fDist_round = 10.0 * np.around ( fDist / 10.0, decimals = 0 )
        sLabelText = "$\mathrm{P}_{2.5}$; SDS011-Sensor ca. %.0f m entfernt" % ( fDist_round )
        ListLabels.append ( sLabelText )
    
    print ( aDistTop3, aKeysTop3 )
    ## "45011", "3815", "15966" , "56599", "3867"
    ## 442.01157292, 496.37905166, 529.40930444, 557.04202662, 690.69178924
    
    aDateTimeLUA, aDataLUA, DictInfo = ds.ImportSensorDataMatterLUA ( sMonthYear = sMonthYear )
    ListDataAverage.append ( ( aDataLUA[ :, 3 ] ) )
    aDateTimeAll, aDataAll, aID, DictInfo, DictDays = ds.LoadRawData ( sDataType = "matter", sDate = None, sSubFolderMonthYear = sMonthYear, 
                                                                       bShowInfo = True )
    
    for sSensorID in aKeysTop3:
        #tAvailableDays = DictDays.get ( sSensorID )
        #print ( tAvailableDays )
        tInfo = DictInfo.get ( sSensorID )
        aSelectID = ( aID == tInfo[ 0 ] )
        aDateTimeNN = aDateTimeAll[ aSelectID ]
        aDataNN = aDataAll[ aSelectID ]
        aDateTimeNN_avg, aDataNN_avg = ds.CalcAverages ( aDateTime = aDateTimeNN, aData = aDataNN, sStartDate = "2026-04-01", 
                                                         sStartTime = "00:00", iWindowDelta = 1, sDeltaUnit = "hours" )
        ListDataAverage.append ( ( aDataNN_avg[ :, 1 ] ) ) ### Mittelwert Pm2.5

    sTitleText = "Einstündige Mittelwerte: April 2026; Nähe %s" % ( DictLoc.get ( sKeyRefLUA ) )
    GraCon = pl.CGraphicConfig ( sTitle = sTitleText, sLabelX = "Zeit", sLabelY = "$\mathrm{P}_{2.5}$ bzw. $\mathrm{PM}_{2.5}$" )
    
    pl.PlotX4Y ( aX = aDateTimeNN_avg, tY = ListDataAverage, tStylesY =
                 ( ( "o12", "D", 3.0, "--", 2.0, "$\mathrm{PM}_{2.5}$; Messstation Landshuter Allee" ),
                   ( "b10", "o", 3.0, "--", 2.0, ListLabels[ 0 ] ),
                   ( "c10", "o", 3.0, "--", 2.0, ListLabels[ 1 ] ),
                   ( "s10", "o", 3.0, "--", 2.0, ListLabels[ 2 ] ) ), GraphicConfig = GraCon )

    return

