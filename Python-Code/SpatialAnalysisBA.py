# -*- coding: utf-8 -*-
# Version vom 21. Juli 2026

import numpy as np
import matplotlib.pyplot as plt
import PlotBA as pl
import gstools # Suffix GsT oder _gst
from sklearn.model_selection import KFold, LeaveOneOut
from gstools import SRF
from gstools import covmodel
from gstools.variogram import vario_estimate_structured
from gstools import krige
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from skgstat.interfaces.gstools import skgstat_to_gstools
import skgstat as sg # Suffix SkG oder _skg
from tabulate import tabulate
from UtilitiesBA import CheckAssert, RescaleImageFile, ConvertRGBImageArrayToGray, SampleFromData2D
from termcolor import colored
from itertools import product
#from RegressionAnalysis import OLSRegression#, CRegression

import warnings
warnings.filterwarnings ( "ignore" )


GiRandSeed = 21339

### ****************** Es gilt für schwach stationäre Zufalls-Felder ********************
## Variogram: 
##  \gamma(r) = \sigma^2 - \sigma^2 * \rho(r) + n
##          \sigma^2 : Varianz des Zufalls-Feldes
##          \rho: Korrelations-Funktion des ZF
##          n: Nugget Effekt
##  Cov(r) = \sigma^2 * rho(r); wobei Cov(r) die Kovarianz-Funktion bezeichnet 

# --------------------------------------------------- Anfang Klasse CVariogramSkG ----------------------------------------------------------
# ******************************* Initialisierung eines Variogramms der Library Scikit-GStat als Klasse ************************************
## Die Daten müssen als äquidistante Matrix übergeben werden, das heißt X_jk ist der Wert an der Postion (j, k)
class CVariogramSkG ( object ):
    def __init__ ( self, aData, sEstimator = "matheron", sModel = "stable", bUseNugget = False, iNumLags = 20, fMaxLag = 10000, 
                   sColorMap = None, iRandSeed = GiRandSeed ):
        self.sEstimator = CheckValidEstimatorSkG ( sEstimator )
        self.sModel = CheckValidCovModelSkG ( sModel )
        CheckAssert ( bBool = ( ( isinstance ( aData, np.ndarray ) ) and ( aData.ndim == 2 ) ), sMsg = "Invalid Type for <aData>!" )
        if ( aData.shape[ 1 ] == 3 ):   
            self.aCoords = np.asarray ( aData[ :, : 2 ], dtype = np.float64 ) 
            self.aData = np.asarray ( aData[ :, 2 ], dtype = np.float64 ) 
            self.aData2D = None
        else:
            self.aCoords = np.asarray ( tuple ( product ( np.arange ( start = 0, stop = aData.shape[ 0 ] ), np.arange ( start = 0, stop = aData.shape[ 1 ] ) ) ) ) 
            self.aData = np.ravel ( aData )
            self.aData2D = aData
            
        print ( colored ( text = ">> CVariogramSkG > Data Shape: %s, Coords Shape: %s" % ( str ( self.aData.shape ), str ( self.aCoords.shape ) ), color = "magenta" ) )
        
        CheckAssert ( bBool = ( ( np.count_nonzero ( np.isnan ( self.aData ) ) == 0 ) or ( np.count_nonzero ( np.isnan ( self.aCoords ) ) ) ), 
                      sMsg = "<aData> and/or <aCoords> contains NA-Values!" )
        self.bUseNugget = bUseNugget
        self.iNumLags = iNumLags
        self.fMaxLag = fMaxLag   # float
        
        self.sFitMethod = "trf"
        self.sBinFunc = "uniform"
        self.DefaultAnnotationFontSize = 12

        if ( sColorMap is not None ):
            self.sColorMap = sColorMap
        else:
            self.sColorMap = "RdYlBu"
        self.CRnG = np.random.default_rng ( seed = iRandSeed ) 
        
        self.aX = None
        self.aBinCenter = None
        self.aNumperBin = None
        self.aVariogram_estd = None
        self.ListBinInfo = None
        self.aVariogram_theo = None
        self.fRho = None
        self.fMSE = None
        self.aResiduals = None
        self.fDefaultNoShape = -999.9
        #self.AIC = None
        #self.BIC = None
        self.fRange = None
        self.fSill =  None
        self.fShape = None
        self.fNugget = None
        self.bShowFitResultTable = True
        self.ListFitBounds = ( [ 3000.0, 3.0, 0.0 ], [ 40000.0, 20.0, 4.5 ] ) ## für P10 ( [ 2000.0, 5.5, 0.0 ], [ 20000.0, 16.0, 6.0 ] ) 
        self.tFitP0 = ( 4000.0, 5.0, 1.5 )
        self.tGsToolsParameter = None

        self.CVariogram_skg = sg.Variogram ( coordinates = self.aCoords, values = self.aData, estimator = self.sEstimator, 
                                             model = self.sModel, dist_func = "euclidean", bin_func = self.sBinFunc, normalize = False, 
                                             fit_method = None, fit_sigma = None, use_nugget = self.bUseNugget, maxlag = self.fMaxLag, 
                                             samples = None, n_lags = self.iNumLags )
        
        if ( sColorMap is not None ):
            self.EstimateVariogram ( tParameterEstimator = ( self.sEstimator, self.sBinFunc, self.fMaxLag ),
                                     ListVariogramEstimation = None, sModel = self.sModel, bUseBounds = False, bShowFit = False )
        
        return
    
    def UpdateModel ( self ):
        CVariogram_skg = sg.Variogram ( coordinates = self.aCoords, values = self.aData, estimator = self.sEstimator, 
                                        model = self.sModel, dist_func = "euclidean", bin_func = self.sBinFunc, normalize = False, 
                                        fit_method = None, fit_sigma = None, use_nugget = self.bUseNugget, maxlag = self.uMaxLag, 
                                        samples = None, n_lags = self.iNumLags )
        
        return ( CVariogram_skg )
    
    def UpdateFitBounds ( self ):
        self.ListFitBounds[ 1 ][ 0 ] = self.uMaxLag
        return
        
    ## Die Routine to_gstools () ist NICHT sinnvoll zu verwenden, da diese nochmals einen Fit durchführt !
    ## skgstat_to_gstools übersetzt mehr oder minder 1:1
    ### from skgstat.interfaces.gstools import skgstat_to_gstools
    def TransformParamtetersToGsTools ( self ):
        try:
            ## schmutziger Workaround, da GsTool einen nu Wert >= 0.2 erfordert für das Matern Modell
            if ( self.sModel == "matern" ):
                CCovModel_gst = skgstat_to_gstools ( self.CVariogram_skg, nu = max ( 0.2, self.fShape ) )
                tParameterTupel = ( CCovModel_gst.var, CCovModel_gst.len_scale, CCovModel_gst.nugget, CCovModel_gst.rescale, 
                                    max ( 0.2, self.fShape ) )
            elif ( self.sModel == "stable" ):
                CCovModel_gst = skgstat_to_gstools ( self.CVariogram_skg, alpha = self.fShape )
                tParameterTupel = ( CCovModel_gst.var, CCovModel_gst.len_scale, CCovModel_gst.nugget, CCovModel_gst.rescale, self.fShape ) 
            else:
                CCovModel_gst = skgstat_to_gstools ( self.CVariogram_skg ) 
                tParameterTupel = ( CCovModel_gst.var, CCovModel_gst.len_scale, CCovModel_gst.nugget, CCovModel_gst.rescale, None )
        except ValueError as sError:
            CCovModel_gst = None
            tParameterTupel = None
            print ( colored ( text = ">> ExportCCovModel_gst > %s" % ( sError ), color = "red", attrs = [ "bold" ] ) )

        self.tGsToolsParameter = tParameterTupel
            
        return 
    
    def GetEstimationParameter ( self, CVariogram_skg ):
        self.aBinCenter, self.aVariogram_estd = CVariogram_skg.get_empirical ( bin_center = True )
        self.aNumPerBin = np.zeros ( shape = ( self.aBinCenter.shape[ 0 ], ), dtype = np.int32 )
        ListBinInfo = list ()
        iCounter = 0
        for ik, aLags in enumerate ( CVariogram_skg.lag_classes () ):
            self.aNumPerBin[ ik ] = aLags.shape[ 0 ]
            iCounter += self.aNumPerBin[ ik ]
            ListBinInfo.append ( ( "(" + str ( self.aNumPerBin[ ik ] ) + ")", self.aBinCenter[ ik ], self.aVariogram_estd[ ik ],
                                 self.DefaultAnnotationFontSize ) )
            
        CheckAssert ( bBool = ( iCounter == np.sum ( self.aNumPerBin ) ), sMsg = "Validation Error!" )
        CheckAssert ( bBool = ( self.aBinCenter.shape == self.aVariogram_estd.shape ), sMsg = "Validation Error!" )
        
        self.ListBinInfo = ListBinInfo
        #print ( self.iNumLags, self.aBinCenter.shape, self.CVariogram_skg.n_lags )
        
        if ( self.iNumLags != self.CVariogram_skg.n_lags ):
            #print ( ">> GetEstimationParameter > #Lags adapted from %d to %d, Binfunc: %s" % ( self.iNumLags, self.CVariogram_skg.n_lags, self.sBinFunc ) )
            self.iNumLags = self.CVariogram_skg.n_lags
        
        return
        
    def ParseParameter ( self, uParameter ):
        tTuple = None
        if ( isinstance ( uParameter, str ) ):
            self.sEstimator = CheckValidEstimatorSkG ( uParameter )
            tTuple = ( self.sEstimator, self.sBinFunc, self.iNumLags, self.uMaxLag ) 
        elif ( isinstance ( uParameter, ( tuple, list, np.ndarray ) ) ):
            if ( len ( uParameter ) == 2 ): 
                self.sEstimator = CheckValidEstimatorSkG ( uParameter[ 0 ] )
                
                if ( isinstance ( uParameter[ 1 ], ( int, np.int_ ) ) ):
                    self.sBinFunc = "even"
                    self.iNumLags = uParameter[ 1 ]
                else:    
                    CheckAssert ( bBool = ( uParameter[ 1 ] in [ "even", "uniform", "sturges", "scott", "fd", "ward" ] ), 
                                  sMsg = "Invalid Format <uParameter!" )
                    self.sBinFunc = uParameter[ 1 ]
                    
                tTuple = ( self.sEstimator, self.sBinFunc, self.iNumLags, self.uMaxLag )
            elif ( len ( uParameter ) == 3 ):
                self.sEstimator = CheckValidEstimatorSkG ( uParameter[ 0 ] )
                
                if ( isinstance ( uParameter[ 1 ], ( int, np.int_ ) ) ):
                    self.sBinFunc = "even"
                    self.iNumLags = uParameter[ 1 ]
                else:    
                    CheckAssert ( bBool = ( uParameter[ 1 ] in [ "even", "uniform", "sturges", "scott", "fd", "ward" ] ), 
                                  sMsg = "Invalid Format <uParameter!" )
                    self.sBinFunc = uParameter[ 1 ]
                
                CheckAssert ( bBool = ( isinstance ( uParameter[ 2 ], ( str, int, float, np.int_, np.float32, np.float64 ) ) ), 
                              sMsg = "Invalid Format <uParameter!", sExtraInfo = "%s" % ( str ( uParameter[ 2 ] ) ) )
                self.uMaxLag = uParameter[ 2 ]
                tTuple = ( self.sEstimator, self.sBinFunc, self.iNumLags, self.uMaxLag )
         
        CheckAssert ( bBool = ( tTuple is not None ), sMsg = "Conversion failed!" )
        
        return ( tTuple )
                      
    def ParseParameterSet ( self, tParameterSet ):
        iNumSets = len ( tParameterSet )
        CheckAssert ( bBool = ( iNumSets in [ 1, 2, 3 ] ), sMsg = "Invalid Shape <tParameterSet>!" )
        
        ListParameter = list ()
        
        if ( isinstance ( tParameterSet[ 0 ], ( list, tuple ) ) ):
            for ik in range ( iNumSets ):
                uParameter = tParameterSet[ ik ]
                tParameterTuple = self.ParseParameter ( uParameter )
                ListParameter.append ( tParameterTuple )
        else:
            CheckAssert ( bBool = ( ( isinstance ( tParameterSet[ 0 ], ( str, np.str_ ) ) ) and 
                                    ( isinstance ( tParameterSet[ 1 ], ( str, np.str_, int, np.int_ ) ) ) and 
                                    ( isinstance ( tParameterSet[ 2 ], ( str, np.str_, int, np.int_, float, np.float32, np.float64 ) ) ) ),
                          sMsg = "Wrong Parameter Format!" )

            tParameterTuple = self.ParseParameter ( tParameterSet )
            ListParameter.append ( tParameterTuple )

        return ( ListParameter )
            
    def GetParameters ( self ):
        ListFitParameter = list ()
        
        self.fRho = self.CVariogram_skg.r
        self.aResiduals = self.CVariogram_skg.residuals
        self.fMSE = np.mean ( np.square ( self.aResiduals ) )
        #self.AIC = self.CVariogram_skg.aic
        #self.BIC = self.CVariogram_skg.bic
        
        tParameter = self.CVariogram_skg.parameters
        
        if ( len ( tParameter ) == 3 ):
            self.fShape = self.fDefaultNoShape
            self.fRange = tParameter[ 0 ]
            self.fSill = tParameter[ 1 ]
            self.fNugget = tParameter[ 2 ]
            tFitParameter = ( self.sModel.capitalize ()[ : 3 ], self.sEstimator.capitalize ()[ : 3 ], self.sBinFunc.lower ()[ : 3 ],
                              "%.0f" % ( self.uMaxLag ), "%.2f" % ( self.fRange ), "%.2f" % ( self.fSill ), 
                              "%.2f" % ( self.fNugget ), "%.3f" % ( self.fMSE ), "%.1f" % ( self.fRho ) )
            ListFitParameter.append ( tFitParameter ) 
            ListHeaders = [ "Mod.", "Est", "Bin", "MaxR", "Ran.", "Sill", "Nug.", "MSE", "Rho" ]
            tColAlign = ( "center", "center", "center", "center", "center", "center", "center", "center", "center" ) 
        elif ( len ( tParameter ) == 4 ):  
            self.fRange = tParameter[ 0 ]
            self.fSill = tParameter[ 1 ]
            self.fShape = tParameter[ 2 ]
            self.fNugget = tParameter[ 3 ]
            tFitParameter = ( self.sModel.capitalize ()[ : 3 ], self.sEstimator.capitalize ()[ : 3 ], self.sBinFunc.lower ()[ : 3 ],
                              "%.0f" % ( self.uMaxLag ), "%.2f" % ( self.fRange ), "%.2f" % ( self.fSill ), 
                              "%.2f" % ( self.fShape ), "%.2f" % ( self.fNugget ), "%.3f" % ( self.fMSE ), "%.1f" % ( self.fRho ) )
            ListFitParameter.append ( tFitParameter ) 
            ListHeaders = [ "Mod.", "Est.", "Bin", "MaxR", "Ran.", "Sill", "Shape", "Nug.", "MSE", "Rho" ]
            tColAlign = ( "center", "center", "center", "center", "center", "center", "center", "center", "center", "center" ) 
        else:
            print ( ">> GetParameters > Something went wrong!" )
            
        if ( self.bShowFitResultTable == True ):
            print ( tabulate ( tabular_data = ListFitParameter, headers = ListHeaders, tablefmt = "pretty", floatfmt = "%.3E", colalign = tColAlign ) )
         
        return
        
    def Fit ( self, sModel, bUseBounds, iNumData = 200, bShowFit = False, bExportParameter = False ):
        self.sModel = CheckValidCovModelSkG ( sModel )
        self.aX = np.linspace ( start = 0.0, stop = self.CVariogram_skg.maxlag, num = iNumData )
        self.CVariogram_skg.model = self.sModel
        
        self.bUseNugget = True
        self.CVariogram_skg.use_nugget = self.bUseNugget
        self.CVariogram_skg.fit_sigma = "sqrt"
        if ( bUseBounds == True ):
            self.UpdateFitBounds ()
            try:
                self.CVariogram_skg.fit ( force = True, method = self.sFitMethod, p0 = self.tFitP0,
                                          bounds = self.ListFitBounds )
            except ( RuntimeError, ZeroDivisionError ) as sError:
                print ( colored ( text = ">> Fit > %s" % ( sError ), color = "red", attrs = [ "bold" ] ) )
                return ( False )
        else:
            self.CVariogram_skg.fit ( force = True, method = self.sFitMethod )             
                                    
        self.GetEstimationParameter ( CVariogram_skg = self.CVariogram_skg ) 
        self.aVariogram_theo = self.CVariogram_skg.fitted_model ( self.aX )
        
        self.GetParameters ()
        if ( bExportParameter == True ):
            self.TransformParamtetersToGsTools ()
        
        if ( bShowFit == True ):
            sTitleText = "$r_{max}$: %.0f, Rg.: %.0f, Sill: %.1f, #Lags: %d, Mdl.: %s, Est.: %s"  % ( self.uMaxLag, self.fRange, self.fSill, 
                                                                                                      self.iNumLags, self.sModel.capitalize ()[ : 3 ], 
                                                                                                      self.sEstimator.capitalize () )
            GraCon = pl.CGraphicConfig ( sTitle = sTitleText, sLabelX = "Distanz $r$", sLabelY = "$\gamma(r)$" )
        
            pl.Plot2X2Y ( aX1 = self.aX, aY1 = self.aVariogram_theo, aX2 = self.aBinCenter, aY2 = self.aVariogram_estd, 
                          tStyleY1 = ( "o12", "o", 0.0, "-", 3.0, "Fit" ), tStyleY2 = ( "b12", "o", 8.0, "--", 2.0, "$\\gamma$" ), GraphicConfig = GraCon )
            
            """
            sTitleText =  "Residuals Plot (Model: %s)" % ( self.sModel.capitalize () )
            CHLine = pl.CLine ( sLineColor = "b12", fLinePos = self.fMSE, fLineWidth = 2.0, sLineLabel = "MSE" )
            CGraCon = pl.CGraphicConfig ( sTitle = sTitleText, sLabelX = "Lags", sLabelY = "$\\left(y_{obs} - y_{theo}\\right)$", HLine1 = CHLine )
            pl.PlotXY ( aX = np.arange ( start = 1, stop = self.aResiduals.shape[ 0 ] + 1 ), aY = self.aResiduals, 
                        tStyle = ( "o12", "o", 6.0, "--", 2.0, "Residuals" ), GraphicConfig = CGraCon )
            """
            
        return ( True )
    
    def ScreenEstimationParameter ( self, tNumLags, tMaxLags, tModels, iShowEachFit = None ):
        CheckAssert ( bBool = ( ( len ( tNumLags ) == 2 ) and ( tNumLags[ 0 ] < tNumLags[ 1 ] ) ), sMsg = "Invalid Shape <tNumLags>!",
                      sExtraInfo = "%s" % ( str ( tNumLags ) ) ) 
        CheckAssert ( bBool = ( ( len ( tMaxLags ) == 3 ) and ( tMaxLags[ 0 ] < tMaxLags[ 1 ] ) ), sMsg = "Invalid Shape <tMaxLag>!",
                      sExtraInfo = "%s" % ( str ( tMaxLags ) ) ) 
        
        self.bShowFitResultTable = False

        aMaxLag = np.linspace ( start = tMaxLags[ 0 ], stop = tMaxLags[ 1 ], num = tMaxLags[ 2 ] )
        aNumLags = np.arange ( start = tNumLags[ 0 ], stop = tNumLags[ 1 ] + 1 )
        ListVariogramEstimation = list ()
        
        iCounter = 0
        for sModel in tModels:
            for fMaxLag in aMaxLag:
                for iNumLags in aNumLags:
                    tParameterEstimator = ( ( "cressie", iNumLags, fMaxLag ), ( "matheron", iNumLags, fMaxLag ) )
                    if ( iShowEachFit is not None ):
                        bShowFit = ( ( iCounter % iShowEachFit ) == 0 )
                    self.EstimateVariogram ( tParameterEstimator = tParameterEstimator, ListVariogramEstimation = ListVariogramEstimation, 
                                             sModel = sModel, bUseBounds = True, bShowFit = bShowFit )
                    iCounter += 1
                    
                
        return ( ListVariogramEstimation )
    
    def EstimateVariogram ( self, tParameterEstimator, ListVariogramEstimation, sModel, bUseBounds, bShowFit ):
        ## tParameter = ( sEstimator, sBinFunc, iNumLags, fMaxLag )
        ListParameter = self.ParseParameterSet ( tParameterSet = tParameterEstimator )
        iNumEstimator = len ( ListParameter )
     
        for ik in range ( iNumEstimator ):
            sEstimator, sBinFunc, iNumLags, fMaxLag = ListParameter[ ik ]
            #print ( sEstimator, sBinFunc, iNumLags, uMaxLag )
            
            self.iNumLags = int ( iNumLags )
            self.CVariogram_skg.n_lags = self.iNumLags
            self.sEstimator = CheckValidEstimatorSkG ( sEstimator )
            self.CVariogram_skg.estimator = self.sEstimator
            self.fMaxLag = fMaxLag
            self.CVariogram_skg.maxlag = self.uMaxLag
            
            self.sBinFunc = sBinFunc
            self.CVariogram_skg.set_bin_func = self.sBinFunc
            ## Schmuztiger Workaround, da die Änderung der Bin-Func NICHT zu einer
            ## Neuberechnung der Bins führt !
            self.CVariogram_skg = self.UpdateModel () 
            self.GetEstimationParameter ( CVariogram_skg = self.CVariogram_skg )
            
            ## es empfieht sich, solche Modelle zu wählen, die keinen zusätzlichen alpha (Stable, Rational) oder 
            ## nu Parameter (Matern) haben und bei denen der rescale == 1 ist 
            ## nach Probieren;  am besten nur ein Kovarianzmodell verwenden: "spherical"
            bCheck = self.Fit ( sModel = sModel, bUseBounds = bUseBounds, iNumData = 300, bExportParameter = False, bShowFit = bShowFit )
            if ( ( ListVariogramEstimation is not None ) and ( bCheck == True ) ):
                ListVariogramEstimation.append ( ( sModel, self.sEstimator.capitalize (), "%d" % ( self.iNumLags ), "%.1f" % ( self.uMaxLag ), 
                                                   "%.3f" % ( self.fRange ), "%.3f" % ( self.fSill ), "%.3f" % ( self.fNugget ), 
                                                   "%.3f" % ( self.fShape ), "%.6f" % ( self.fMSE ) ) )
                
        return
        
    def ShowVariogramEstimation ( self, tParameterEstimator, sColorMap = "RdYlBu_r", sImageTitle = "Daten" ):
        ## tParameter = ( sEstimator, sBinFunc, iNumLags, uMaxLag )
        tStyleEstimation = ( ( "o12", "o", 6.0, "-", 2.0 ), ( "b9", "o", 6.0, "-", 2.0 ), ( "c12", "o", 6.0, "-", 2.0 ) )
        ListParameter = self.ParseParameterSet ( tParameterSet = tParameterEstimator )
        #print ( ListParameter )
        
        iNumEstimator = len ( ListParameter )
        ListSemivarianceEstimation = list ()
        ListStyleEstimation = list ()
        ListTitle = list ()
        ListBinInfos = list ()
            
        for ik in range ( iNumEstimator ):
            self.iNumLags = ListParameter[ ik ][ 2 ]
            self.CVariogram_skg.n_lags = self.iNumLags
            self.sEstimator = CheckValidEstimatorSkG ( ListParameter[ ik ][ 0 ] )
            self.CVariogram_skg.estimator = self.sEstimator
            self.uMaxLag = ListParameter[ ik ][ 3 ]
            self.CVariogram_skg.maxlag = self.uMaxLag
            
            self.sBinFunc = ListParameter[ ik ][ 1 ]
            self.CVariogram_skg.set_bin_func = self.sBinFunc
            ## Schmuztiger Workaround, da die Änderung der Bin-Func NICHT zu einer
            ## Neuberechnung der Bins führt !
            self.CVariogram_skg = self.UpdateModel () 
            
            self.GetEstimationParameter ( CVariogram_skg = self.CVariogram_skg )
               
            sTitleText = "Variogramm (%s)" % ( self.sEstimator.capitalize () )
            #sLabel = "$\hat{\gamma}(r)\ (r_{max}=%.0f,\,N=%d,\,%s)$" % ( self.uMaxLag, self.iNumLags, self.sBinFunc[ : 4 ] )
            sLabel = "$\hat{\gamma}(r)\ (r_{max}=%.0f,\,N=%d)$" % ( self.uMaxLag, self.iNumLags )
            ListTitle.append ( sTitleText )
            sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth = tStyleEstimation[ ik ]
            tStyleEstimationComplete = ( sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, sLabel )
            ListStyleEstimation.append ( tStyleEstimationComplete )
        
            """ ### debug
            print ( ">>>>> 2", self.CVariogram_skg.maxlag, self.uMaxLag )          
            self.CVariogram_skg.set_bin_func = "even"#ListParameter[ ik ][ 1 ] 
            self.CVariogram_skg.estimator = self.sEstimator
            self.CVariogram_skg.preprocessing ( force = True) 
            print ( self.CVariogram_skg.get_empirical ( bin_center = True )[ 0 ] )
            print ( ">>>>> 3", self.CVariogram_skg.maxlag, self.uMaxLag )
            """
            aSemivarianceEstimation = np.zeros ( shape = ( self.iNumLags, 2 ), dtype = np.float64 )
            aSemivarianceEstimation[ :, 0 ] = self.aBinCenter
            
            aSemivarianceEstimation[ :, 1 ] = self.CVariogram_skg.experimental
            ListSemivarianceEstimation.append ( aSemivarianceEstimation )
            ListBinInfos.append ( self.ListBinInfo )
            ## Ende for Schleife
            
        CGraCon = pl.CGraphicConfig ( sTitle = ListTitle[ 0 ], sLabelY = "$\hat{\gamma}(r)$", 
                                      sLabelX = "Abstand $r$ (m)", sLabelX2 = "Abstand $r$ (m)", 
                                      sLabelX3 = "Abstand $r$ (m)", sGridAxis = "both", sStepPlotWhere = "none" )
        if ( iNumEstimator == 1 ):
            aSemiVariance1 = ListSemivarianceEstimation[ 0 ]
            pl.PlotXY ( aX = aSemiVariance1[ :, 0 ], aY = aSemiVariance1[ :, 1 ], tStyle = ListStyleEstimation[ 0 ],
                        GraphicConfig = CGraCon, ListAnnotation = ListBinInfos[ 0 ] )
        if ( iNumEstimator == 2 ):
            CGraCon.Set ( sTitle2 = ListTitle[ 1 ] )
            aSemiVariance1 = ListSemivarianceEstimation[ 0 ]
            aSemiVariance2 = ListSemivarianceEstimation[ 1 ]
            pl.PlotStackHorz2X2Y ( aX1 = aSemiVariance1[ :, 0 ], aY1 = aSemiVariance1[ :, 1 ], 
                                   aX2 = aSemiVariance2[ :, 0 ], aY2 = aSemiVariance2[ :, 1 ], GraphicConfig = CGraCon, 
                                   tStyles = ( ListStyleEstimation[ 0 ], ListStyleEstimation[ 1 ] ),
                                   ListAnnotations = ListBinInfos, bShareY = False )
     
        if ( iNumEstimator == 3 ):
            CGraCon.Set ( sTitle2 = ListTitle[ 1 ], sTitle3 = ListTitle[ 2 ] )
            aSemiVariance1 = ListSemivarianceEstimation[ 0 ]
            aSemiVariance2 = ListSemivarianceEstimation[ 1 ]
            aSemiVariance3 = ListSemivarianceEstimation[ 2 ]
            pl.PlotStackHorz3X3Y ( aX1 = aSemiVariance1[ :, 0 ], aY1 = aSemiVariance1[ :, 1 ], 
                                   aX2 = aSemiVariance2[ :, 0 ], aY2 = aSemiVariance2[ :, 1 ], 
                                   aX3 = aSemiVariance3[ :, 0 ], aY3 = aSemiVariance3[ :, 1 ], GraphicConfig = CGraCon, 
                                   tStyles = ( ListStyleEstimation[ 0 ], ListStyleEstimation[ 1 ], ListStyleEstimation[ 2 ] ),
                                   ListAnnotations = ListBinInfos, bShareY = False ) 

        if ( self.aData2D is not None ):
            CGraCon.Set ( sTitle = sImageTitle, sLabelX = "X", sLabelY = "Y" )
            sLabel = "$\\hat{\gamma}$ (%s)" % ( ListParameter[ 0 ][ 0 ].capitalize () ) 
            tStyleEstimation1 = ( "o12", "o", 4.0, "", 0.0, sLabel )
            
            pl.PlotStackHorzImageXY ( aX = aSemiVariance1[ :, 0 ], aData = aSemiVariance1[ :, 1 ], aData2Dim = self.aData2D, 
                                       GraphicConfig = CGraCon, tStyle = tStyleEstimation1, sColorMap = sColorMap, sOrigin = "upper" )
    
        return 

    def CompareCovModel4 ( self, iNumData = 200 ):
        tStyleEstimation = ( "o12", "o", 6.0, "", 0.0, "Estimation" )
        tStyleTheory = ( "b12", "", 0.0, "--", 2.0, "Theory" )
        tKernel = ( "spherical", "exponential", "cubic", "matern" )
        ## fMSE, fRange, fSill, fNugget, fShape
        aRank = np.zeros ( shape = ( len ( tKernel ), 6 ), dtype = np.float64 )
        
        CFigure, tCAxis = plt.subplots ( ncols = 2, nrows = 2, figsize = ( 13, 8 ), sharex = True, sharey = False )
        aCAxes = tCAxis.flatten ()
        self.CVariogram_skg.maxlag = self.uMaxLag
        self.CVariogram_skg.n_lag = self.iNumLags
        self.CVariogram_skg.set_bin_func = self.sBinFunc
        self.CVariogram_skg = self.UpdateModel () 
        
        self.bShowFitResultTable = True 
        
        CGraCon = pl.CGraphicConfig ( sLabelX = "Abstands-Klasse $r$", sLabelY = "$\hat{\gamma}(r)$" )
        for ik, sModel in enumerate ( tKernel ):
            self.Fit ( sModel = sModel, bUseBounds = True, iNumData = iNumData, bExportParameter = True )
            #print ( self.tGsToolsParameter )
            
            sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, sLabel = tStyleTheory
            aCAxes[ ik ].plot ( self.aX, self.aVariogram_theo, color = pl.GetColor ( sColor ) , marker = sMarker, markersize = fMarkerSize, 
                                linestyle = sLineStyle, linewidth = fLineWidth, label = sLabel )
            sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, sLabel = tStyleEstimation
            aCAxes[ ik ].plot ( self.aBinCenter, self.aVariogram_estd, color = pl.GetColor ( sColor ) , marker = sMarker, markersize = fMarkerSize, 
                                linestyle = sLineStyle, linewidth = fLineWidth, label = sLabel )
            
            fVar, fLenScale, fNugget, fRescale, fShape = self.tGsToolsParameter
            aRank[ ik ] = self.fMSE, fVar, fLenScale, fNugget, fRescale, fShape

            sTitleText =  "Modell: %s, Schätzer: %s (MSE: %.2E)" % ( sModel.capitalize (), self.sEstimator.capitalize (),  self.fMSE )
            CHLine1 = pl.CLine ( sLineColor = "g15", fLinePos = self.fSill + self.fNugget, fLineWidth = 1.5, sLineStyle = "-" )
            CHLine2 = pl.CLine ( sLineColor = "g10", fLinePos = self.fNugget, fLineWidth = 1.5, sLineStyle = "-" )
            CVLine1 = pl.CLine ( sLineColor = "c9", fLinePos = self.fRange, fLineWidth = 1.5, sLineStyle = "-." )
            CGraCon.Set ( sTitle = sTitleText, VLine1 = CVLine1, HLine1 = CHLine1 )
            pl.DrawFrameAxis ( CAxis = aCAxes[ ik ], GraphicConfig = CGraCon, iIndex = 0, sGridAxis = "both", 
                               tHLines = ( CHLine2, ), tSizes = ( 14, 14, 12 ) )
            
        plt.tight_layout ()
        plt.show ()
        
        aIndicesRankModel = np.argsort ( aRank[ :, 0 ] )
        aRank = aRank[ aIndicesRankModel ]
        ListRankEstimator = [ tKernel[ iIdx ] for iIdx in aIndicesRankModel ] 
       
        return ( aRank, ListRankEstimator )
    
    def CompareEstimator ( self, sModel, iNumLags, iNumData = 100, tEstimator = None ):
        tStyleEstimation = ( "o9", "o", 5.0, "", 0.0, "Estimation" )
        tStyleTheory = ( "b12", "", 0.0, "--", 2.0, "Theory" )
        if ( tEstimator is None ):
            tEstimator = ( "matheron", "cressie", "dowd" )
        else:
            CheckAssert ( bBool = ( len ( tEstimator ) == 3 ), sMsg = "Length <tEstimator> must be three!" )
        
        self.sModel = CheckValidCovModelSkG ( sModel )
        self.iNumLags = iNumLags
        self.CVariogram_skg.n_lags = self.iNumLags

        CFigure, tCAxis = plt.subplots ( ncols = 3, nrows = 1, figsize = ( 12, 4 ), sharex = True, sharey = False )
        aCAxes = tCAxis.flatten ()
        
        CGraCon = pl.CGraphicConfig ( sLabelX = "Abstand $r$ (m)", sLabelY = "$\hat{\gamma}(r)$" )
        for ik, sEstimator in enumerate ( tEstimator ):
            self.CVariogram_skg.estimator = sEstimator
            self.CVariogram_skg.n_lags = self.iNumLags
            
            aX, aVariogram_theo, aBinCenter, aVariogram_estd = self.Fit ( sModel = self.sModel, iNumData = iNumData, bShowResiduals = False )
            sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, sLabel = tStyleTheory
            aCAxes[ ik ].plot ( aX, aVariogram_theo, color = pl.GetColor ( sColor ) , marker = sMarker, markersize = fMarkerSize, 
                                linestyle = sLineStyle, linewidth = fLineWidth, label = sLabel )
            sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, sLabel = tStyleEstimation
            aCAxes[ ik ].plot ( aBinCenter, aVariogram_estd, color = pl.GetColor ( sColor ) , marker = sMarker, markersize = fMarkerSize, 
                                linestyle = sLineStyle, linewidth = fLineWidth, label = sLabel )
            
            sTitleText =  "Estimator: %s (Model: %s); MSE: %.1E" % ( sEstimator.capitalize (), self.sModel.capitalize (), self.fMSE )
            CHLine = pl.CLine ( sLineColor = "g12", fLinePos = self.fSill, fLineWidth = 1.0 )
            CVLine = pl.CLine ( sLineColor = "g12", fLinePos = self.fRange, fLineWidth = 1.0 )
            CGraCon.Set ( sTitle = sTitleText, VLine1 = CVLine, HLine1 = CHLine )
            pl.DrawFrameAxis ( CAxis = aCAxes[ ik ], GraphicConfig = CGraCon, iIndex = 0, sGridAxis = "both", tSizes = ( 10, 10, 8 ) )
            
        plt.tight_layout ()
        plt.show ()
        
        return
# --------------------------------------------------- Ende Klasse CVariogramSkG ------------------------------------------------------------

# --------------------------------------------------- Anfang Klasse COrdKrigingGsT ---------------------------------------------------------
# ***************************** Initialisierung eines Ordinary Kriging Modells der Library GsTools als Klasse ******************************
class COrdKrigingGsT ( object ):
    def __init__ ( self, uDataObserved, CCovarianceModel = None, CCovarianceModel_gst = None, bFitVariogram = False ):
        if ( CCovarianceModel is not None ):
            CheckAssert ( bBool = ( isinstance ( CCovarianceModel, CCovarianceModelGsT ) ), 
                          sMsg = "Wrong Paramter Class" )
            self.CCovarianceModel_gst = CCovarianceModel.CCovModel_gst
        if ( CCovarianceModel_gst is not None ):
            CheckAssert ( bBool = ( ( isinstance ( CCovarianceModel_gst, ( gstools.covmodel.models.Gaussian ) ) ) or
                                    ( isinstance ( CCovarianceModel_gst, ( gstools.covmodel.models.Exponential ) ) ) or
                                    ( isinstance ( CCovarianceModel_gst, ( gstools.covmodel.models.Matern ) ) ) or
                                    ( isinstance ( CCovarianceModel_gst, ( gstools.covmodel.models.Rational ) ) ) or
                                    ( isinstance ( CCovarianceModel_gst, ( gstools.covmodel.models.Spherical ) ) ) or
                                    ( isinstance ( CCovarianceModel_gst, ( gstools.covmodel.models.Cubic ) ) ) or
                                    ( isinstance ( CCovarianceModel_gst, ( gstools.covmodel.models.Stable ) ) ) ), 
                          sMsg = "Wrong Paramter Class" )
            self.CCovarianceModel_gst = CCovarianceModel_gst
     
        CheckAssert ( bBool = ( ( isinstance ( uDataObserved, np.ndarray ) and ( uDataObserved.shape [ 0 ] == 3 ) ) or
                                ( isinstance ( uDataObserved, ( list, tuple ) ) and ( len ( uDataObserved ) == 3 ) ) ), 
                                  sMsg = "Invalid Shape <uDataObserved>!" )
        if ( isinstance ( uDataObserved, ( list, tuple ) ) ):
            self.aDataObserved = np.asarray ( uDataObserved )
        else:
            self.aDataObserved = uDataObserved
            
        CheckAssert ( bBool = ( self.aDataObserved.shape[ 0 ] != 0 ), sMsg = "Inavlid Shape <aDataObserved>!" )

        self.CCovarianceModel = CCovarianceModel   
        self.bFitVariogram = bFitVariogram
        
        ## pseudo_inv_type = "pinvh" ist etwa Faktor drei schneller als "pinv"
        self.COrdinaryKrige_gst = krige.Ordinary ( model = self.CCovarianceModel_gst, 
                                                   cond_pos = [ self.aDataObserved[ 0, : ], self.aDataObserved[ 1, : ] ], 
                                                   cond_val = self.aDataObserved[ 2, : ], normalizer = None, fit_normalizer = False, 
                                                   fit_variogram = self.bFitVariogram, pseudo_inv = True, pseudo_inv_type = "pinvh", 
                                                   exact = True )
        
        return
    
    def Interpolate ( self, aGridX = None, aGridY = None, tDimX = None, tDimY = None, tLimX = None, tLimY = None, 
                      iNumLevel = None, sColorMap = "RdYlBu_r", CInfoBox = None, bShowInfo = True, bShowVariance = True ):
        CheckAssert ( bBool = ( ( ( aGridX is not None ) and ( aGridY is not None ) ) or 
                                ( ( tDimX is not None ) and ( tDimY is not None ) ) ), sMsg = "Initialization of Grid Failed!" )
        
        if ( ( tDimX is not None ) and ( tDimY is not None ) ):
            aGridX = np.linspace ( start = tDimX[ 0 ], stop = tDimX[ 1 ], num = tDimX[ 2 ] )
            aGridY = np.linspace ( start = tDimY[ 0 ], stop = tDimY[ 1 ], num = tDimY[ 2 ] )
            
        if ( bShowInfo == True ):
            print ( colored ( text = ">> COrdKrigingGsT > Interpolate: GridX: %s, GridY: %s" % ( str ( aGridX.shape ), str ( aGridY.shape ) ), 
                              color = "magenta" ) )
            print ( colored ( text = ">> COrdKrigingGsT > Interpolate: Model: %s" % ( str ( self.COrdinaryKrige_gst.model ) ), color = "magenta" ) ) 
            
        aRField = np.asarray ( self.COrdinaryKrige_gst.structured ( [ aGridX, aGridY ], return_var = True ), dtype = np.float64 )

        if ( bShowInfo == True ):
            #print ( self.COrdinaryKrige_gst )
            print ( ">> Mean: %.3f" % ( self.COrdinaryKrige_gst.get_mean ( post_process = True ) ) )
            
        if ( iNumLevel is not None ):
            if ( self.CCovarianceModel is not None ):
                sTitleText = "Ordinary Kriging (" + self.CCovarianceModel.CCovModel_gst.name.capitalize ()[ : 3] + ". Korr.-Fkt.): "
            else:
                sTitleText = "Ordinary Kriging (" + self.CCovarianceModel_gst.name.capitalize ()[ : 3 ] + ". Korr.-Fkt.): "

            ShowRegressionResult ( aGridX = aGridX, aGridY = aGridY, aZ_mean = aRField[ 0 ].T, aZ_var = aRField[ 1 ].T, aZ = None, 
                                   aDataObserved = self.aDataObserved, sTitleText = sTitleText, sColorMap = sColorMap, 
                                   iNumLevel = iNumLevel, tLimX = tLimX, tLimY = tLimY, CInfoBox = CInfoBox, bShowVariance = bShowVariance )
            
        return ( aRField )
    
    def Predict ( self, uPos, aDataObserved = None, bShowInfo = True ):
        fMSE = None
        CheckAssert ( bBool = ( ( isinstance ( uPos, np.ndarray ) and ( uPos.shape [ 0 ] == 2 ) ) or
                                ( isinstance ( uPos, ( list, tuple ) ) and ( len ( uPos ) == 2 ) ) ), 
                                  sMsg = "Invalid Shape <uDataObserved>!" )
        if ( isinstance ( uPos, ( list, tuple ) ) ):
            aPos = np.asarray ( uPos )
        else:
            aPos = uPos
            
        if ( bShowInfo == True ):
            print ( colored ( text = ">> COrdKrigingGsT > Predict: aPos: %s" % ( str ( aPos.shape ) ), color = "magenta" ) )
            print ( colored ( text = ">> COrdKrigingGsT > Predict: Model: %s" % ( str ( self.COrdinaryKrige_gst.model ) ), color = "magenta" ) ) 

        aRField = np.asarray ( self.COrdinaryKrige_gst.unstructured ( pos = [ aPos[ 0, : ], aPos[ 1, : ] ], return_var = False ), dtype = np.float64 )
        
        if ( aDataObserved is not None ):
            fMSE = np.mean ( np.square ( aRField - aDataObserved ) )
            
        return ( fMSE )
    
    def RunCrossValidation ( self, iNumFolds, bShowInfo = True ):
        CheckAssert ( bBool = ( iNumFolds <= self.aDataObserved.shape[ 1 ] ), sMsg = "Invalid Choice for <iNumFolds>!" )
        
        aMSE = np.zeros ( shape = ( iNumFolds, ), dtype = np.float32 )
        if ( iNumFolds < self.aDataObserved.shape[ 1 ] ): 
            CCrossValidation = KFold ( n_splits = iNumFolds, shuffle = True, random_state = 21336  ) 
        else:
            CCrossValidation = LeaveOneOut ()
            
        for ik, ( aIndices_train, aIndices_test ) in enumerate ( CCrossValidation.split ( np.transpose ( self.aDataObserved ) ) ):
            if ( bShowInfo == True ):
                print ( ">> COrdKrigingGsT > RunCrossValidation: Model: %s" % ( str ( self.COrdinaryKrige_gst.model ) ) ) 
            self.COrdinaryKrige_gst.set_condition ( cond_pos = [ self.aDataObserved[ 0, aIndices_train ], self.aDataObserved[ 1, aIndices_train ] ], 
                                                    cond_val = self.aDataObserved[ 2, aIndices_train ], fit_normalizer = False, 
                                                    fit_variogram = self.bFitVariogram )
            #print ( self.COrdinaryKrige_gst.model ) 
            fMSE = self.Predict ( uPos = [ self.aDataObserved[ 0, aIndices_test ], self.aDataObserved[ 1, aIndices_test ] ], 
                                  aDataObserved = self.aDataObserved[ 2, aIndices_test ], bShowInfo = False )
             
            ## es sind insbesondere die Ausreißer, die die fMSE Werte richtig nach oben ziehen
            aMSE[ ik ] = fMSE

        return ( aMSE )
# -------------------------------------------------- Ende Klasse COrdKriging_GsTools -------------------------------------------------------
    
# --------------------------------------------------- Anfang Klasse CCovModelGsT -----------------------------------------------------------
# ******************************** Initialisierung eines Kovarianz-Modells der Library GsTools als Klasse **********************************
class CCovarianceModelGsT ( object ):
    sDescription_long : str = ""
    sDescription : str = ""
    sDescription_short : str = ""
    sDescription_compact : str = ""
    fNugget : float = 0.0
    
    fVar : float = None
    CCovModel_gst : gstools.covmodel.models = None
    aRandomField : np.ndarray = None
   
    def __init__ ( self, sModel, fVar = 1.0, uLenScale = 1.0, fShape = None, fAngle = 0.0, fNugget = 0.0, fRescale = 1.0,
                   bLatLon = False, bInfo = True ):
        sModel = sModel.upper ()
        
        CheckAssert ( bBool = ( sModel in [ "GAUSSIAN", "MATERN", "EXPONENTIAL", "RATIONAL", "SPHERICAL", "STABLE", "CUBIC" ] ), 
                      sMsg = "Invalid Parameter <sModel>!" )
        
        CheckAssert ( bBool = ( isinstance ( uLenScale, ( float, np.float32, np.float64, tuple, list, np.ndarray ) ) ), 
                      sMsg = "Invalid Parameter <uLenScale>!" )
            
        if ( sModel == "GAUSSIAN" ):
            ## Parameter: var, len_scale, angles, nugget 
            self.CCovModel_gst = covmodel.Gaussian ( dim = 2, latlon = bLatLon, geo_scale = gstools.tools.KM_SCALE, 
                                                     var = fVar, len_scale = uLenScale, angles = fAngle, nugget = fNugget )
        elif ( sModel == "EXPONENTIAL" ):
            ## Parameter: var, len_scale, angles, nugget
            self.CCovModel_gst = covmodel.Exponential ( dim = 2, latlon = bLatLon, geo_scale = gstools.tools.KM_SCALE, 
                                                        var = fVar, len_scale = uLenScale, angles = fAngle, nugget = fNugget )            
        elif ( sModel == "SPHERICAL" ):
            ## Parameter: var, len_scale, angles, nugget
            self.CCovModel_gst = covmodel.Spherical ( dim = 2, latlon = bLatLon, geo_scale = gstools.tools.KM_SCALE, 
                                                      var = fVar, len_scale = uLenScale, angles = fAngle, nugget = fNugget )      
        elif ( sModel == "MATERN" ):
            ## Parameter: var, len_scale, nu, angles, rescale, nugget
            CheckAssert ( bBool = ( fShape is not None ), sMsg = "<fShape> must be greater Zero!" )
            self.CCovModel_gst = covmodel.Matern ( dim = 2, latlon = bLatLon, geo_scale = gstools.tools.KM_SCALE, 
                                                   var = fVar, len_scale = uLenScale, nu = fShape, 
                                                   rescale = fRescale, angles = fAngle, nugget = fNugget )    
        elif ( sModel == "RATIONAL" ):
            ## Parameter: var, len_scale, alpha, angles, nugget    
            CheckAssert ( bBool = ( fShape is not None ), sMsg = "<fShape> must be greater Zero!" )
            self.CCovModel_gst = covmodel.Rational ( dim = 2, latlon = bLatLon, geo_scale = gstools.tools.KM_SCALE, 
                                                     var = fVar, len_scale = uLenScale, alpha = fShape, angles = fAngle, nugget = fNugget )      
        elif ( sModel == "STABLE" ):
            ## Parameter: var, len_scale, alpha, angles, rescale, nugget
            CheckAssert ( bBool = ( fShape is not None ), sMsg = "<fShape> must be greater Zero!" )
            self.CCovModel_gst = covmodel.Stable ( dim = 2, latlon = bLatLon, geo_scale = gstools.tools.KM_SCALE, 
                                                   var = fVar, len_scale = uLenScale, alpha = fShape, 
                                                   rescale = fRescale, angles = fAngle, nugget = fNugget )
        elif ( sModel == "CUBIC" ):
            ## Parameter: var, len_scale, angles, nugget
            self.CCovModel_gst = covmodel.Cubic ( dim = 2, latlon = bLatLon, geo_scale = gstools.tools.KM_SCALE, 
                                                  var = fVar, len_scale = uLenScale, angles = fAngle, nugget = fNugget )
        else:
            print ( "failure" )
            
        self.sDescription_long, self.sDescription, self.sDescription_short, self.sDescription_compact = GetDescription ( self.CCovModel_gst ) 
        self.fVar = fVar
        self.fNugget = fNugget
        if ( bInfo == True ):
            self.GetModelDescription ()
        
        return
        
    def GetModelDescription ( self ):
        print ( ">> CCovarianceModelGsT > Model : %s" % ( str ( self.CCovModel_gst ) ) )

        return
    
# +++++++++++++++++++++++++++++++++++++++++++++++++ Darstellung des Verlaufs der Korrelation +++++++++++++++++++++++++++++++++++++++++++++++
## Variogram: 
##  \gamma(r) = \sigma^2 - \sigma^2 * \rho(r) + n
##          \sigma^2 : Varianz des Zufalls-Feldes
##          \rho: Korrelations-Funktion des ZF
##          n: Nugget Effekt
##  Cov(r) = \sigma^2 * rho(r); wobei Cov(r) die Kovarianz-Funktion bezeichnet 
    def PlotStatistics ( self, fStop = 5.0, iNumPoints = 200 ):
        if ( self.fVar != 1.0 ):
            CGraCon = pl.CGraphicConfig ( sTitle = self.sDescription_long, sLabelX = "Lag $r$", sLabelY = "$\\gamma(r)$, $\\rho(r)$, $\\text{Cov}(r)$" )
            aR = np.linspace ( start = 0.0, stop = fStop, num = iNumPoints )
            aGamma = self.CCovModel_gst.variogram ( r = aR )
            aRho = self.CCovModel_gst.correlation ( r = aR )
            aCov = self.CCovModel_gst.covariance ( r = aR )

            pl.PlotX3Y ( aX = aR, aY1 = aGamma, aY2 = aRho, aY3 = aCov,
                         tStyleY1 = ( "c10", "x", 0.0, "--", 3.0, "$\\gamma$" ),
                         tStyleY2 = ( "b10", "D", 0.0, "-.", 3.0, "$\\rho$" ), 
                         tStyleY3 = ( "r14", "o", 0.0, "-", 3.0, "$\\text{Cov}$" ), GraphicConfig = CGraCon )
        else:
            CGraCon = pl.CGraphicConfig ( sTitle = self.sDescription_long, sLabelX = "Distanz $r$", sLabelY = "$\\gamma(r)$, $\\rho(r)$" )
            aR = np.linspace ( start = 0.0, stop = fStop, num = iNumPoints )
            aGamma = self.CCovModel_gst.variogram ( r = aR )
            aRho = self.CCovModel_gst.correlation ( r = aR )

            pl.PlotX2Y ( aX = aR, aY1 = aGamma, aY2 = aRho,
                         tStyleY1 = ( "c10", "x", 0.0, "-.", 3.0, "$\\gamma$" ),
                         tStyleY2 = ( "b10", "D", 0.0, "-.", 3.0, "$\\rho$" ), GraphicConfig = CGraCon )
            
        return
    
    def PlotVariogram ( self, fStop, tLengthScale, iNumPoints = 200 ):
        CheckAssert ( bBool = ( len ( tLengthScale ) == 3 ), sMsg = "Invalid Shape <tLenghtScale>" )
        
        self.sDescription_long, self.sDescription, self.sDescription_short, self.sDescription_compact = GetDescription ( self.CCovModel_gst, tLengthScale, "gamma" ) 
        CGraCon = pl.CGraphicConfig ( sTitle = self.sDescription_long, sLabelX = "Lag $r$", sLabelY = "Semivarianz $\\gamma(r)$" )
            
        aR = np.linspace ( start = 0.0, stop = fStop, num = iNumPoints )
        ListGamma = list ()
        for fLengthScale in tLengthScale:
            self.CCovModel_gst.len_scale = fLengthScale
            self.CCovModel_gst.nugget = self.fNugget
            
            aGamma = self.CCovModel_gst.variogram ( r = aR )
            sLabel = "$\\gamma$ für $\ell=%.1f$" % fLengthScale
            ListGamma.append ( ( aGamma, sLabel ) )

        pl.PlotX3Y ( aX = aR, aY1 = ListGamma[ 0 ][ 0 ], aY2 = ListGamma[ 1 ][ 0 ], aY3 = ListGamma[ 2 ][ 0 ],
                     tStyleY1 = ( "c10", "x", 0.0, "--", 3.0, ListGamma[ 0 ][ 1 ] ),
                     tStyleY2 = ( "b10", "D", 0.0, "-.", 3.0, ListGamma[ 1 ][ 1 ] ), 
                     tStyleY3 = ( "r14", "o", 0.0, "-", 3.0, ListGamma[ 2 ][ 1 ] ), GraphicConfig = CGraCon )

        return
    
    def PlotCorrelogram ( self, fStop, tLengthScale, iNumPoints = 200 ):
        CheckAssert ( bBool = ( len ( tLengthScale ) == 3 ), sMsg = "Invalid Shape <tLenghtScale>" )
        
        self.sDescription_long, self.sDescription, self.sDescription_short, self.sDescription_compact = GetDescription ( self.CCovModel_gst, tLengthScale, "rho" ) 
        CGraCon = pl.CGraphicConfig ( sTitle = self.sDescription_long, sLabelX = "Lag $r$", sLabelY = "Korrelation $\\rho(r)$" )
            
        aR = np.linspace ( start = 0.0, stop = fStop, num = iNumPoints )
        ListRho = list ()
        for fLengthScale in tLengthScale:
            self.CCovModel_gst.len_scale = fLengthScale
            self.CCovModel_gst.nugget = self.fNugget
            aRho = self.CCovModel_gst.correlation ( r = aR )
            sLabel = "$\\rho$ für $\ell=%.1f$" % fLengthScale
            ListRho.append ( ( aRho, sLabel ) )

        pl.PlotX3Y ( aX = aR, aY1 = ListRho[ 0 ][ 0 ], aY2 = ListRho[ 1 ][ 0 ], aY3 = ListRho[ 2 ][ 0 ],
                     tStyleY1 = ( "c10", "x", 0.0, "--", 3.0, ListRho[ 0 ][ 1 ] ),
                     tStyleY2 = ( "b10", "D", 0.0, "-.", 3.0, ListRho[ 1 ][ 1 ] ), 
                     tStyleY3 = ( "r14", "o", 0.0, "-", 3.0, ListRho[ 2 ][ 1 ] ), GraphicConfig = CGraCon )

        return
    
    def PlotStack_GRF_Variogram ( self, tDimX, tDimY, fLengthScale, fMean = 0.0, sColorMap = "RdYlBu", iRandSeed = GiRandSeed ):
        CheckAssert ( bBool = ( len ( tDimX ) == 3 ), sMsg = "Invalid Shape <tDimX>!" )
        CheckAssert ( bBool = ( len ( tDimY ) == 3 ), sMsg = "Invalid Shape <tDimY>!" )

        aX_rf = np.linspace ( start = tDimX[ 0 ], stop = tDimX[ 1 ], num = tDimX[ 2 ] )
        aY_rf = np.linspace ( start = tDimY[ 0 ], stop = tDimY[ 1 ], num = tDimY[ 2 ] )
        
        aR = np.linspace ( start = 0.0, stop = min ( tDimX[ 1 ], tDimY[ 1 ] ), num = min ( tDimX[ 2 ], tDimY[ 2 ] ) )
        self.CCovModel_gst.len_scale = fLengthScale
        self.CCovModel_gst.nugget = self.fNugget
        self.sDescription_long, self.sDescription, self.sDescription_short, self.sDescription_compact = GetDescription ( CCovModel_gst = self.CCovModel_gst, 
                                                                                                                          tLengthScale = None, sType = "gamma" )
        aGamma = self.CCovModel_gst.variogram ( r = aR )
        
        CSRF = SRF ( model = self.CCovModel_gst, mean = fMean, seed = iRandSeed )
        self.aRandomField = CSRF ( ( aX_rf, aY_rf ), mesh_type = "structured" )
        
        if ( fMean != 0.0 ):
            sDescription = "Gauß ZF für $\mu = %.1f,\,$"  % ( fMean )
        else:
            sDescription = "Gauß ZF für "
        sDescription = sDescription + self.sDescription
        
        CGraCon = pl.CGraphicConfig ( sTitle = sDescription, sLabelX = "X", sLabelY = "Y", 
                                      sTitle2 = "Variogramm", sLabelX2 = "Lag r", sLabelY2 = "$\gamma(r)$", sGridAxis = "both" )
        pl.PlotStackHorzImageXY ( aX = aY_rf, aData = aGamma, aData2Dim = self.aRandomField, GraphicConfig = CGraCon, 
                                  tStyle = ( "c10", "x", 0.0, "--", 3.0, "$\\gamma$" ), sColorMap = sColorMap, 
                                  tExtent = ( tDimX[ 0 ], tDimX[ 1 ], tDimY[ 0 ], tDimY[ 1 ] ), sOrigin = "lower" )
        
        return
    
    def PlotStack_GRF_Correlogram ( self, tDimX, tDimY, fLengthScale, fMean = 0.0, sColorMap = "RdYlBu", iRandSeed = GiRandSeed ):
        CheckAssert ( bBool = ( len ( tDimX ) == 3 ), sMsg = "Invalid Shape <tDimX>!" )
        CheckAssert ( bBool = ( len ( tDimY ) == 3 ), sMsg = "Invalid Shape <tDimY>!" )

        aX_rf = np.linspace ( start = tDimX[ 0 ], stop = tDimX[ 1 ], num = tDimX[ 2 ] )
        aY_rf = np.linspace ( start = tDimY[ 0 ], stop = tDimY[ 1 ], num = tDimY[ 2 ] )
        
        aR = np.linspace ( start = 0.0, stop = min ( tDimX[ 1 ], tDimY[ 1 ] ), num = min ( tDimX[ 2 ], tDimY[ 2 ] ) )
        self.CCovModel_gst.len_scale = fLengthScale
        self.CCovModel_gst.nugget = self.fNugget
        self.sDescription_long, self.sDescription, self.sDescription_short, self.sDescription_compact = GetDescription ( CCovModel_gst = self.CCovModel_gst,
                                                                                                                          tLengthScale = None, sType = "rho" )
        aRho = self.CCovModel_gst.correlation ( r = aR )
        
        CSRF = SRF ( model = self.CCovModel_gst, mean = fMean, seed = iRandSeed )
        self.aRandomField = CSRF ( ( aX_rf, aY_rf ), mesh_type = "structured" )
        
        if ( fMean != 0.0 ):
            sDescription = "Gauß ZF für $\mu = %.1f,\,$"  % ( fMean )
        else:
            sDescription = "Gauß ZF für "
        sDescription = sDescription + self.sDescription + "$,\,\sigma^2=%.1f$" % ( self.CCovModel_gst.var )
        
        CGraCon = pl.CGraphicConfig ( sTitle = sDescription, sLabelX = "X", sLabelY = "Y", 
                                      sTitle2 = "Korrelogramm", sLabelX2 = "Lag r", sLabelY2 = "$\\rho(r)$", sGridAxis = "both" )
        pl.PlotStackHorzImageXY ( aX = aY_rf, aData = aRho, aData2Dim = self.aRandomField, GraphicConfig = CGraCon, 
                                  tStyle = ( "c10", "x", 0.0, "--", 3.0, "$\\rho$" ), sColorMap = sColorMap, 
                                  tExtent = ( tDimX[ 0 ], tDimX[ 1 ], tDimY[ 0 ], tDimY[ 1 ] ), sOrigin = "lower" )
        
        return
        
# --------------------------------------------------- Ende Klasse CCovModelGsT -------------------------------------------------------------

# ------------------------------------------------ Anfang Klasse CRandomFieldGsT -----------------------------------------------------------
# **** Generierung und Analyse eines zufälligen räumlichen Feldes mit vorgegebenem Kovarianz Modell anhand der Library GsTools ***
class CRandomFieldGsT ( object ):
    sDescription : str = ""
    fMean : float = None
    iDimX : int = None
    iDimY : int = None
    CCovModel : CCovarianceModelGsT = None
    aX_rf : np.ndarray = None
    aY_rf : np.ndarray = None
    aGammaX : np.ndarray = None
    aVariogramX_theo : np.ndarray = None
    aGammaY : np.ndarray = None
    aVariogramY_theo : np.ndarray = None
    ListVarioStyleX : list = [ "g12", "o", 5.0, "-", 2.0, "$\gamma_x$" ]
    ListVarioStyleY : list = [ "b12", "o", 5.0, "-", 3.0, "$\gamma_y$" ]
    ListVarioStyleFit : list = [ "o12", "", 0.0, "-.", 3.0, "$f_x$" ] 
    aRandomField : np.ndarray = None
    
    def __init__ ( self, tDim, sModel, fVar = 1.0, uLenScale = 1.0, fShape = None, fNu = None, fAngles = 0.0, fNugget = 0.0, 
                   fMean = 0.0, iRandSeed = GiRandSeed ):
        CheckAssert ( bBool = ( len ( tDim ) == 2 ), sMsg = "Invalid Shape <tDim>!" )
        self.CCovModel = CCovarianceModelGsT ( sModel = sModel, fVar = fVar, uLenScale = uLenScale, fShape = fShape, 
                                               fAngles = fAngles, fNugget = fNugget )
        self.fMean = fMean
        self.iDimX = tDim[ 0 ]
        self.iDimY = tDim[ 1 ]
        self.aX_rf = np.arange ( start = 0, stop = self.iDimX )
        self.aY_rf = np.arange ( start = 0, stop = self.iDimY )
        CSRF = SRF ( model = self.CCovModel.CCovModel_gst, mean = self.fMean, seed = iRandSeed )
        self.aRandomField = CSRF ( ( self.aX_rf, self.aY_rf ), mesh_type = "structured" )
    
        self.sDescription = "$\mathcal{GRF}\,[$" + self.CCovModel.sDescription_compact + "]"
        
        return
    
    def EstimateVariogram ( self, sColorMap = "RdYlBu" ):
        self.aGammaX = vario_estimate_structured ( self.aRandomField, direction = "x", estimator = "cressie" )
        self.aGammaY = vario_estimate_structured ( self.aRandomField, direction = "y", estimator = "cressie" )
        
        if ( sColorMap ):
            CGraCon = pl.CGraphicConfig ( sTitle = "Variogramm X-Richtung", sLabelX = "X", sLabelY = "$\gamma_x$", 
                                       sTitle2 = "Variogramm Y-Richtung", sLabelX2 = "Y", sLabelY2 = "$\gamma_y$", sGridAxis = "both" )
            pl.PlotStackHorz2X2Y ( aX1 = self.aX_rf[ : 200 ], aY1 = self.aGammaX[ : 200 ], aX2 = self.aY_rf[ : 200 ], aY2 = self.aGammaY[ : 200 ], 
                                   tStyles = ( self.ListVarioStyleX, self.ListVarioStyleY ), GraphicConfig  = CGraCon )
    
            CGraCon.Set ( sTitle = self.sDescription, sLabelY = "Y" )
            pl.PlotStackHorzImageXY ( aX = self.aY_rf, aData = self.aGammaY, aData2Dim = self.aRandomField, GraphicConfig = CGraCon, 
                                      tStyle = self.ListVarioStyleY, sColorMap = sColorMap, sOrigin = "lower" )

        return
    
# ++++++++++++++++++++++++++++++ Fit eines theoretischen Kovarianz-Modells an die beobachteten Daten +++++++++++++++++++++++++++++++++++++++
    def FitVariogram ( self, sDirection, tArgBounds = None, bShowInfo = True ):
        sDirection = sDirection.upper ()
        CheckAssert ( bBool = ( sDirection in [ "X", "Y", "BOTH", "XY" ] ), sMsg = "Invalid Parameter <sDirection>!" )
        if ( ( tArgBounds is not None ) and ( len ( tArgBounds ) == 2 ) ):
            self.CCovModel.CCovModel_gst.set_arg_bounds ( var = tArgBounds[ 0 ], len_scale = tArgBounds[ 1 ] )
            
        if ( ( tArgBounds is not None ) and ( len ( tArgBounds ) == 3 ) ):
            if ( isinstance ( self.CCovModel.CCovModel_gst, gstools.covmodel.models.Matern ) ):
                self.CCovModel.CCovModel_gst.set_arg_bounds ( var = tArgBounds[ 0 ], len_scale = tArgBounds[ 1 ], nu = tArgBounds[ 1 ] )
            elif ( isinstance ( self.CCovModel.CCovModel_gst, gstools.covmodel.models.Rational ) ):
                self.CCovModel.CCovModel_gst.set_arg_bounds ( var = tArgBounds[ 0 ], len_scale = tArgBounds[ 1 ], alpha = tArgBounds[ 1 ] )
                
        if ( ( self.aGammaX is None ) or ( self.aGammaY is None ) ):
            self.EstimateVariogram ( sColorMap = "" )
            
        ## X-Richtung
        if ( sDirection in [ "X", "XY", "BOTH" ] ):
            DictResultX, pcov, fR2X = self.CCovModel.CCovModel_gst.fit_variogram ( self.aX_rf, self.aGammaX, nugget = False, return_r2 = True )
            DictResultX[ "R2" ] = fR2X
            self.aVariogramX_theo = self.CCovModel.CCovModel_gst.variogram ( r = self.aX_rf )
            
            if ( bShowInfo == True ):
                self.ListVarioStyleX[ 5 ] = "obs"
                _, sText, _, _ = GetDescription ( CCovModel_gst = self.CCovModel.CCovModel_gst )
                sTitleText = sText + " $(\\text{R}^2: %.3f)$" % ( DictResultX.get ( "R2" ) )
                CGC = pl.CGraphicConfig ( sTitle = "Fit X-Direction > " + sTitleText, sLabelX = "r", sLabelY = "Variogram" )
                pl.PlotX2Y ( aX = self.aX_rf, aY1 = self.aGammaX, aY2 = self.aVariogramX_theo, GraphicConfig = CGC, 
                             tStyleY1 = self.ListVarioStyleX, tStyleY2 = self.ListVarioStyleFit )
        
        ## Y-Richtung
        if ( sDirection in [ "Y", "XY", "BOTH" ] ):
            DictResultY, pcov, fR2Y = self.CCovModel.CCovModel_gst.fit_variogram ( self.aY_rf, self.aGammaY, nugget = False, return_r2 = True )
            DictResultY[ "R2" ] = fR2Y
            self.aVariogramY_theo = self.CCovModel.CCovModel_gst.variogram ( r = self.aY_rf )

            if ( bShowInfo == True ):
                self.ListVarioStyleY[ 5 ] = "obs"
                self.ListVarioStyleFit[ 5 ] = "$f_y$"
                _, sText, _, _ = GetDescription ( CCovModel_gst = self.CCovModel.CCovModel_gst )
                sTitleText = sText + " $(\\text{R}^2: %.3f)$" % ( DictResultY.get ( "R2" ) )
                CGC = pl.CGraphicConfig ( sTitle = "Fit Y-Direction > " + sTitleText, sLabelX = "r", sLabelY = "Variogram" )
                pl.PlotX2Y ( aX = self.aY_rf, aY1 = self.aGammaY, aY2 = self.aVariogramY_theo, GraphicConfig = CGC, 
                             tStyleY1 = self.ListVarioStyleY, tStyleY2 = self.ListVarioStyleFit )

        if ( bShowInfo == True ):
            print ( DictResultX, DictResultY )
        
        return 
    
    def CheckIsotropy ( self ):
        if ( ( self.aVariogramX_theo is None ) or ( self.aVariogramY_theo is None ) ):
            self.FitVariogram ( sDirection = "both", tArgBounds = None, bShowInfo = False )
            
        iMin = self.aVariogramX_theo.shape[ 0 ]
        if ( iMin != self.aVariogramY_theo.shape[ 0 ] ):
            iMin = min ( self.aVariogramX_theo.shape[ 0 ], self.aVariogramY_theo.shape[ 0 ] )
            
        aLine = np.linspace ( start = 0.0, stop = 1.0, num = self.aX_rf.shape[ 0 ] )
        
        CGC = pl.CGraphicConfig ( sTitle = "Isotropie Check", sLabelX = "theo. Semivarianz x-Direction", 
                                  sLabelY = "theo. Semivarianz y-Direction" )
        pl.Plot2X2Y ( aX1 = self.aVariogramX_theo[ : iMin ], aY1 = self.aVariogramY_theo[ : iMin ], aX2 = aLine, aY2 = aLine, 
                      GraphicConfig = CGC, tStyleY1 = ( "s9", "o", 5.0, "--", 2.0, "theo. Variograms" ),
                      tStyleY2 = ( "r12", "o", 0.0, "--", 2.0, "\"Line of Isotropy\"" ) )
        
        return
    
    def Show ( self, sColorMap = "RdYlBu", sTitleText = None ):
        sBarLabel = "z(x, y)" 
        if ( sTitleText is None ):
            sTitleText = self.sDescription
        CGC = pl.CGraphicConfig ( sTitle = sTitleText, sLabelX = "X", sLabelY = "Y", sLegend = sBarLabel, sGridAxis = "none" )
        
        if ( self.CCovModel.CCovModel_gst.anis != 1.0 ):
            sBoxText = "$\lambda = %.2f$" % ( self.CCovModel.CCovModel_gst.anis[ 0 ] )
            CInfobox = pl.CInfoBox ( fBoxPosX = 5, fBoxPosY = self.iDimY - 15, sText = sBoxText, sFaceColor = "chartreuse" )
            CGC.Set ( InfoBox = CInfobox )

        pl.PlotImage ( aData2Dim = self.aRandomField, GraphicConfig = CGC, sColorMap = sColorMap, sOrigin = "upper" )
    
        return 
    
# ------------------------------------------------- Ende Klasse CRandomFieldGsT ------------------------------------------------------------

# ************************ Empirische Schätzung des direktionalen Variogramms inkusive Anpassung an Modell *******************************
def EstimateDirectionalVariogram ( aCoords, aData, sModel, sDate, iNumDirections, ListFitBounds = None, sPivotParameter = "sill", 
                                   tTextLabel = None, bUseNugget = True ):
    if ( ListFitBounds is not None ):
        CheckAssert ( bBool = ( ( len ( ListFitBounds ) == 2 ) and ( ( len ( ListFitBounds[ - 1 ] ) in [ 2, 3 ] ) ) ), 
                      sMsg = "Invalid Format <ListFitBounds>!" )
    else:
        if ( bUseNugget == True ):
            ListFitBounds = ( [ 3000.0, 0.5, 0.0 ], [ 22000.0, 20.0, 4.0 ] )
            tFitP0 = ( 10000.0, 3.0, 1.0 )
        else:
            ListFitBounds = ( [ 3000.0, 0.5 ], [ 22000.0, 20.0 ] )
            tFitP0 = ( 10000.0, 6.0 )
            
    iNumData = 300
    sModel = CheckValidCovModelSkG ( sModel )
    DictPivotParameter = { "Range" : 1, "Sill" : 2, "Nugget" : 3 }

    if ( tTextLabel is not None ):
        sAverageParameter, sDescription, sUnit, iNumSensors = tTextLabel
        aDataID = np.arange ( start = 1, stop = iNumSensors + 1 )
        sTitleText = "Kenngröße %s aller %d Sensoren im %s" % ( sAverageParameter.capitalize (), iNumSensors, sDate )
        GraCon = pl.CGraphicConfig ( sTitle = sTitleText, sLabelX = "CountID", sLabelY = sDescription + " (" + sUnit + ")" )
        pl.PlotXY ( aX = aDataID, aY = aData, tStyle = ( "s8", "o", 8.0, "", 0.0, "" ), GraphicConfig = GraCon ) 

    ## aAngles in Grad \beta mit: -180 <= \beta <= 180
    ### The azimuth of the directional dependence for this Variogram, given as an angle in degree. 
    ### The East of the coordinate plane is set to be at 0° and is counted clockwise to 180° and counter-clockwise to -180°.
    aAngles = np.linspace ( start = 0.0, stop = 360.0, num = iNumDirections, endpoint = False )
    
    ListGamma = list ()
    aFitParameter = np.zeros ( shape = ( iNumDirections, 4 ), dtype = np.float64 )

    for ik in range ( 0, iNumDirections ):
        #if ( aAngles[ ik ] <= 180.0 ):
        #    fAngle = aAngles[ ik ]
        #else:
        ## selbes Ergebnis wie bei der if-else Abfrage
        fAngle = aAngles[ ik ] - 180.0
        CDirVario = sg.DirectionalVariogram ( coordinates = aCoords, values = aData, estimator = "cressie", model = sModel, 
                                              dist_func = "euclidean", bin_func = "sturges", normalize = False, fit_method = "trf", 
                                              fit_sigma = None, directional_model = "triangle", azimuth = fAngle, 
                                              tolerance = 45, bandwidth = "q33", use_nugget = bUseNugget, maxlag = None, n_lags = 12, 
                                              verbose = False )
        
        aBinCenter, aGamma = CDirVario.get_empirical ( bin_center = True )
        fMaxLag = np.amax ( aBinCenter ) 

        aCounts = np.zeros ( shape = ( aBinCenter.shape[ 0 ], ), dtype = np.int16 )
        for ij, aLags in enumerate ( CDirVario.lag_classes () ):
            aCounts[ ij ] = aLags.shape[ 0 ]
        
        CDirVario.fit ( force = True, method = "trf", p0 = tFitP0, bounds = ListFitBounds )  
        print ( CDirVario )
        tParameter = CDirVario.parameters
        print ( aAngles[ ik ], tParameter )
        if ( len ( tParameter ) == 4 ):
            fRange, fSill, fNugget = tParameter[ 0 ], tParameter[ 1 ], tParameter[ 3 ]
        else:
            fRange, fSill, fNugget = tParameter[ 0 ], tParameter[ 1 ], tParameter[ 2 ]
            
        aFitParameter[ ik ] = aAngles[ ik ], fRange, fSill, fNugget
        
        aX = np.linspace ( start = 0.0, stop = fMaxLag, num = iNumData )
        aVariogram_theo = CDirVario.fitted_model ( aX )
        ListGamma.append ( ( aGamma, aCounts ) )
        
        ### Check, ob der Sill wirklich der Sill ist oder womöglih der partial Sill
        ## Sill ist der echte Sill also Sill = partial Sill + Nugget
        
        CDirVario.fit ( method = "manual", range = fRange, sill = fSill, nugget = fNugget )
        aVariogram_theo2 = CDirVario.fitted_model ( aX )
        
        print ( np.allclose ( a = aVariogram_theo, b = aVariogram_theo2 ) )
        
        if ( tTextLabel is not None ):
            ListAnnotation = list ()
            for il in range ( aCounts.shape[ 0 ] ):
                ListAnnotation.append ( ( " (%s)" % ( aCounts[ il ] ), aBinCenter[ il ], aGamma[ il ], 10, "s18" ) )
            sFitLabel = "Fit (%s)" % ( sModel.capitalize () )
            CHLine1 = pl.CLine ( sLineColor = "r8", fLinePos = fNugget, sLineStyle = "--", fLineWidth = 2.0, sLineLabel = "Nugget" )
            CHLine2 = pl.CLine ( sLineColor = "r12", fLinePos = fSill, sLineStyle = "-", fLineWidth = 2.0,  sLineLabel = "Sill" )
            CVLine1 = pl.CLine ( sLineColor = "s12", fLinePos = fRange, sLineStyle = "--", fLineWidth = 2.0, sLineLabel = "Range" )
            GraCon.Set ( sTitle = "Anpassung an direktionale Semivarianz ($\\vartheta= %.0f°$); %s" % ( aAngles[ ik ], sDate ), 
                         sLabelX = "Distanz r (m)", sAnnotationHorzAlign = "left", sAnnotationVertAlign = "bottom", HLine1 = CHLine1, HLine2 = CHLine2, VLine1 = CVLine1,
                        )
            pl.Plot2X2Y ( aX1 = aBinCenter, aY1 = aGamma, aX2 = aX, aY2 = aVariogram_theo, 
                          tStyleY1 = ( "c9", "o", 9.0, "", 0.0, "$\gamma(r)$" ), 
                          tStyleY2 = ( "b12", "", 0.0, "--", 3.0, sFitLabel ), ListAnnotation = ListAnnotation, GraphicConfig = GraCon )

    if ( sPivotParameter is not None ):
        sPivotParameter = sPivotParameter.capitalize ()
        iIndex = DictPivotParameter.get ( sPivotParameter )
        sTitleText = "Direktionale Variogramm-Karte (%s); %s" % ( sPivotParameter, sDate ) 
        GraCon = pl.CGraphicConfig ( sTitle = sTitleText, fPosVariable = 270.0 )
        
        #if ( sPivotParameter == "Sill" ):
         #   aY = aFitParameter[ :, iIndex ] + aFitParameter[ :, 2 ]
        #else:
        aY = aFitParameter[ :, iIndex ]
        aRad =  np.deg2rad ( aFitParameter[ :, 0 ] )
        
        #print ( aY )
        #print ( aRad )
        tStyleScatter = ( "o", 40, "b12" )
        
        if ( sPivotParameter == "Sill" ):
            tCrossStyle  = ( 3.0, "o12", 2.0, "o10" )
        else:
            tCrossStyle = None
            
        pl.PlotPolar ( aRad = aRad, aRho = aY, tStyleScatter = tStyleScatter, tCrossStyle = tCrossStyle, GraphicConfig = GraCon ) 
        
    return ( aBinCenter, ListGamma, aFitParameter )

# ************************************************************** Hilfs-Funktionen  *********************************************************
def CheckValidCovModelSkG ( sModel ):
    sModel = sModel.lower () 
    CheckAssert ( bBool = ( sModel in [ "spherical", "exponential", "gaussian", "matern", "stable", "cubic" ] ), 
                  sMsg = "Invalid Choice for <sModel>!", sExtraInfo = sModel )
    
    return ( sModel )

def CheckValidEstimatorSkG ( sEstimator ):
    sEstimator = sEstimator.lower ()
    CheckAssert ( bBool = ( sEstimator in [ "matheron", "cressie", "dowd", "genton", "minmax", "entropy" ] ), 
                  sMsg = "Invalid Choice for <sEstimator>!", sExtraInfo = sEstimator )
    
    return ( sEstimator )
    
# ***************************************** Abfrage der Attribute eines Kovarianz-Modells **************************************************
def GetDescription ( CCovModel_gst, tLengthScale = None, sType = "gamma" ):
    sType = sType.lower ()
    CheckAssert ( bBool = ( sType in [ "gamma", "rho" ] ), sMsg = "Invalid Paramer <sType>!" )
    
    if ( tLengthScale is not None ):
        sTextLengthScale = "\{ %.1f, %.1f, %.1f\}" % ( tLengthScale[ 0 ], tLengthScale[ 1 ], tLengthScale[ 2 ] )
    else:
        sTextLengthScale = "%s" % ( str ( CCovModel_gst.len_scale ) )
    
    if ( isinstance ( CCovModel_gst, ( gstools.covmodel.models.Gaussian ) ) ):
        if ( sType == "gamma" ):
            sDescription = "$\\gamma_{gau}\,\\left(r;\,\ell = %s\\right),\ \sigma^2 = %.1f$" % ( sTextLengthScale, CCovModel_gst.var )
        else:
            sDescription = "$\\rho_{gau}\,\\left(r;\,\ell = %s\\right)$" % ( sTextLengthScale )
        sDescription_long = "Gauß'sches Modell: " + sDescription
        sDescription_short = "$\\%s_{gau}\,\\left(r;\,\ell = %s\\right)$" % ( sType, sTextLengthScale )
        sDescription_compact = "$\\%s_{gau}\,\\left(r\\right)$" % ( sType )
    elif ( isinstance ( CCovModel_gst, gstools.covmodel.models.Exponential ) ):
        if ( sType == "gamma" ):
            sDescription = "$\\gamma_{exp}\,\\left(r;\,\ell = %s\\right),\ \sigma^2 = %.1f$" % ( sTextLengthScale, CCovModel_gst.var )
        else:
            sDescription = "$\\rho_{exp}\,\\left(r;\,\ell = %s\\right)$" % ( sTextLengthScale )
        sDescription_long = "Exponentielles Modell: " + sDescription
        sDescription_short = "$\\%s_{exp}\,\\left(r;\,\ell = %s\\right)$" % ( sType, sTextLengthScale )
        sDescription_compact = "$\\%s_{exp}\,\\left(r\\right)$" % ( sType )
    elif ( isinstance ( CCovModel_gst, gstools.covmodel.models.Matern ) ):
        if ( sType == "gamma" ):
            sDescription = "$\\gamma_{mat}\,\\left(r;\,\ell = %s,\ \\nu = %.1f\\right),\ \sigma^2 = %.1f$" % ( sTextLengthScale, CCovModel_gst.nu, CCovModel_gst.var )
        else:
            sDescription = "$\\rho_{mat}\,\\left(r;\,\ell = %s,\ \\nu = %.1f\\right)$" % ( sTextLengthScale, CCovModel_gst.nu )
        sDescription_long = "Matérn Modell: " + sDescription
        sDescription_short = "$\\%s_{mat}\,\\left(r;\,[\ell = %s,\ \\nu = %.1f\\right)$" % ( sType, sTextLengthScale, CCovModel_gst.nu )
        sDescription_compact = "$\\%s_{mat}\,\\left(r\\right)$" % ( sType )
    elif ( isinstance ( CCovModel_gst, gstools.covmodel.models.Rational ) ):
        if ( sType == "gamma" ):
            sDescription = "$\\gamma_{rat}\,\\left(r;\,\ell = %s,\ \\alpha = %.1f\\right),\ \sigma^2 = %.1f$" % ( sTextLengthScale, CCovModel_gst.alpha, CCovModel_gst.var )
        else:
            sDescription = "$\\rho_{rat}\,\\left(r;\,\ell = %s,\ \\alpha = %.1f\\right)$" % ( sTextLengthScale, CCovModel_gst.alpha )
        sDescription_long = "Rationales Modell: " + sDescription
        sDescription_short = "$\\%sa_{rat}\,\\left(r;\,\ell = %s,\ \\alpha = %.1f\\right)$" % ( sType, sTextLengthScale, CCovModel_gst.alpha )
        sDescription_compact = "$\\%s_{rat}\,\\left(r\\right)$" % ( sType )
    elif ( isinstance ( CCovModel_gst, gstools.covmodel.models.Spherical ) ):
        if ( sType == "gamma" ):
            sDescription = "$\\gamma_{sph}\,\\left(r;\,\ell = %s\\right),\ \sigma^2 = %.1f$" % ( sTextLengthScale, CCovModel_gst.var )
        else:
            sDescription = "$\\rho_{sph}\,\\left(r;\,\ell = %s\\right)$" % ( sTextLengthScale )
        sDescription_long = "Sphärisches Modell: " + sDescription
        sDescription_short = "$\\%s_{sph}\,\\left(r;\,\ell = %s\\right)$" % ( sType, sTextLengthScale )
        sDescription_compact = "$\\%s_{sph}\,\\left(r\\right)$" % ( sType )
    elif ( isinstance ( CCovModel_gst, gstools.covmodel.models.Stable ) ):
        if ( sType == "gamma" ):
            sDescription = "$\\gamma_{stb}\,\\left(r;\,\ell = %s,\ \\alpha = %.1f\\right),\ \sigma^2 = %.1f$" % ( sTextLengthScale, CCovModel_gst.alpha, CCovModel_gst.var ) 
        else:
            sDescription = "$\\rho_{stb}\,\\left(r;\,\ell = %s,\ \\alpha = %.1f\\right)$" % ( sTextLengthScale, CCovModel_gst.alpha ) 
        sDescription_long = "Stabiles Modell: " + sDescription
        sDescription_short = "$\\%s_{stb}\,\\left(r;\,\ell = %s,\ \\alpha = %.1f\\right)$" % ( sType, sTextLengthScale, CCovModel_gst.alpha ) 
        sDescription_compact = "$\\%s_{stb}\,\\left(r\\right)$" % ( sType )
    elif ( isinstance ( CCovModel_gst, gstools.covmodel.models.Cubic ) ):
        if ( sType == "gamma" ):
            sDescription = "$\\gamma_{cub}\,\\left(r;\,\ell = %s\\right),\ \sigma^2 = %.1f$" % ( sTextLengthScale, CCovModel_gst.var )
        else:
            sDescription = "$\\rho_{cub}\,\\left(r;\,\ell = %s\\right)$" % ( sTextLengthScale )
        sDescription_long = "Kubisches Modell: " + sDescription
        sDescription_short = "$\\%s_{cub}\,\\left(r;\,\ell = %s\\right)$" % ( sType, sTextLengthScale )
        sDescription_compact = "$\\%s_{cub}\,\\left(r\\right)$" % ( sType )  
    else:
        print ( ">> Model not supported!" )
              
    return ( sDescription_long, sDescription, sDescription_short, sDescription_compact )

# ************************************************ Custom Kernel Functions using Scikit Learn **********************************************
def BuildKernelRBF ( tParRBF ):
    CheckAssert ( bBool = ( ( len ( tParRBF ) == 2 ) or ( len ( tParRBF ) == 3 ) ), sMsg = "Invalid Parameter Shape!" )
    if ( len ( tParRBF ) == 3 ):
        CKernelRBF  = tParRBF[ 0 ] * kernels.RBF ( length_scale = tParRBF[ 1 ], length_scale_bounds = tParRBF[ 2 ] )  
    else:
        CKernelRBF  = tParRBF[ 0 ] * kernels.RBF ( length_scale = tParRBF[ 1 ], length_scale_bounds = ( 1E-2, 1E5 ) )   
    
    return ( CKernelRBF )

def BuildKernelRational ( tParRational ):
    CheckAssert ( bBool = ( ( len ( tParRational ) == 3 ) ), sMsg = "Invalid Parameter Shape!" )
    CKernelRational = tParRational[ 0 ] * kernels.RationalQuadratic ( length_scale = tParRational[ 1 ], alpha = tParRational[ 2 ] )

    return ( CKernelRational )

def BuildKernelWhiteNoise ( tParWhiteNoise ):
    CheckAssert ( bBool = ( ( len ( tParWhiteNoise ) == 2 ) or ( len ( tParWhiteNoise ) == 3 ) ), sMsg = "Invalid Parameter Shape!" )
    if ( len ( tParWhiteNoise ) == 3 ):
        CKernelWhiteNoise = tParWhiteNoise[ 0 ] * kernels.WhiteKernel ( noise_level = tParWhiteNoise[ 1 ], noise_level_bounds = tParWhiteNoise[ 2 ] )
    else:
        CKernelWhiteNoise = tParWhiteNoise[ 0 ] * kernels.WhiteKernel ( noise_level = tParWhiteNoise[ 1 ], noise_level_bounds = ( 1E-5, 1E5 ) )
    
    return ( CKernelWhiteNoise )

def BuildKernelMatern ( tParMatern ):
    CheckAssert ( bBool = ( ( len ( tParMatern ) == 2 ) or ( len ( tParMatern ) == 3 ) ), sMsg = "Invalid Parameter Shape!" )
    # Nu = 3/2 ist deutlich schlechter!
    if ( len ( tParMatern ) == 3 ):
        CKernelMatern = tParMatern[ 0 ] * kernels.Matern ( length_scale = tParMatern[ 1 ], nu = 2.5, length_scale_bounds = tParMatern[ 2 ] )  
    else: 
        CKernelMatern = tParMatern[ 0 ] * kernels.Matern ( length_scale = tParMatern[ 1 ], nu = 2.5, length_scale_bounds = ( 1E-2, 1E2 ) )  
    
    return ( CKernelMatern )

# *************************** Krigung Demo mittels des Gaussian Process Regressors von Scikit Learn ****************************************
def DemoGaussianProcessRegression2D ( tDimX = ( -4.0, 4.0, 50 ), tDimY = ( -5.0, 5.0, 60 ), fSplit = 0.99, iNumLevel = 20, iRandSeed = 21339 ):
    CheckAssert ( bBool = ( ( len ( tDimX ) == 3 ) and ( len ( tDimY ) == 3 ) ), sMsg = "Invalid Parameter Shape" )
    CRnG = np.random.default_rng ( seed = iRandSeed )
    fNoiseLevel = 3.0
    
    aGridX = np.linspace ( start = tDimX[ 0 ], stop = tDimX[ 1 ], num = tDimX[ 2 ] )  # x-vector
    aGridY = np.linspace ( start = tDimY[ 0 ], stop = tDimY[ 1 ], num = tDimY[ 2 ] )  # y-vector
    aX, aY = np.meshgrid ( aGridX, aGridY )  # x and y are (nv, nv) matrices
    aXY = np.column_stack ( ( np.ravel ( aX ), np.ravel ( aY ) ) )

    aZ = ( aX - 3.0 )**2 + 2.0 * aX * aY + ( 2.0 * aY + 3.0 )**2 - 3.0
    aZ = aZ + CRnG.normal ( size = aX.shape ) * fNoiseLevel
    aZ_flat = np.ravel ( aZ )
    
    CKernel = ( BuildKernelRBF ( tParRBF = ( 1.0, ( 2.0, 1.0 ), ( ( 1E-1, 100.0 ), ) * 2 ) ) + 
                BuildKernelWhiteNoise ( tParWhiteNoise = ( 1.0, 4.0, ( 1E-20, 10.0 ) ) ) )
    
    aXY_train, aXY_test, aZ_train, aZ_test = train_test_split ( aXY, aZ_flat, test_size = fSplit, random_state = iRandSeed )
    aDataObserved = np.vstack ( ( aXY_train[ :, 0 ], aXY_train[ :, 1 ], aZ_train ) )
    
    CGPR = GaussianProcessRegressor ( CKernel, normalize_y = True )
    CGPR.fit ( aXY_train, aZ_train )
    print ( "Kernel after Fit >\n%s" % ( str ( CGPR.kernel_ ) ) )
    aZ_mean, aZ_std = CGPR.predict ( aXY, return_std = True )

    aZ_var = np.square ( np.reshape ( aZ_std, shape = aZ.shape ) )
    aZ_mean = np.reshape ( aZ_mean, shape = aZ_var.shape )
    
    ShowRegressionResult ( aGridX = aGridX, aGridY = aGridY, aZ_mean = aZ_mean, aZ_var = aZ_var, aZ = aZ, 
                           aDataObserved = aDataObserved, sTitleText = "$\mathcal{GP}$ Regression > ", sColorMap = "RdYlBu_r", iNumLevel = iNumLevel )
    
    return
# *************** Routine zur Darstellung von Erwartungswert, Varianz und ggf MSE der Schätzung von Kriging bzw GP Regression **************
def ShowRegressionResult ( aGridX, aGridY, aZ_mean, aZ_var, aZ = None, aDataObserved = None, sTitleText = "", sColorMap = "RdYlBu_r", 
                           iNumLevel = 20, CInfoBox = None, tLimX = None, tLimY = None, bShowVariance = True ):
    tStyleDataObs_mean = ( sColorMap, "o", 80.0, "Observation", "s5" )
    tStyleDataObs_var = ( "s9", "X", 8.0, "Observation", "s9" )
    tStyleDataObs_data = ( "s9", "H", 8.0, "Observation", "s9" )
    
    #print ( aDataObserved )
    #print ( ">> ShowRegressionresult > aGridX Shape: %s, aGridY Shape: %s" % ( str ( aGridX.shape ), str ( aGridY.shape ) ) )
    #print ( ">> ShowRegressionResult > aZ_mean Shape: %s, aZ_var Shape: %s, aData_obs Shape: %s" % ( str ( aZ_mean.shape ), str ( aZ_var.shape ), str ( aDataObserved.shape ) ) )
    CGraCon = pl.CGraphicConfig ( sTitle = "Spatial Data", sLabelX = "x", sLabelY = "y", sLegend = "$f(x, y)$" )

    if ( aZ is not None ):
        pl.PlotContour ( aX = aGridX, aY = aGridY, aData2D = aZ, iNumLevel = iNumLevel, GraphicConfig = CGraCon, sColorMap = sColorMap,
                         tStyleDataObs = ( sColorMap, "x", 12.0, "Z", "s2" ) )
    
    if ( aZ is not None ):
        aMSE = ( np.square ( aZ - aZ_mean ) )
        print ( ">> MSE: %.3f" % ( np.mean ( aMSE ) ) )
     
    CGraCon.Set ( sTitle = sTitleText + "Erwartungswert", sLabelX = "x", sLabelY = "y", sLegend = "Erwartungswert" )
    if ( CInfoBox is not None ):
        CGraCon.Set ( InfoBox = CInfoBox )
        
    pl.PlotContour ( aX = aGridX, aY = aGridY, aData2D = aZ_mean, iNumLevel = iNumLevel, GraphicConfig = CGraCon, sColorMap = sColorMap, 
                     aDataObserved = aDataObserved, tStyleDataObs = tStyleDataObs_mean, tLimX = tLimX, tLimY = tLimY )
    
    if ( bShowVariance == True ):
        CGraCon.Set ( sTitle = sTitleText + "Varianz", sLegend = "Varianz" )
        pl.PlotContour ( aX = aGridX, aY = aGridY, aData2D = aZ_var, iNumLevel = iNumLevel, GraphicConfig = CGraCon, sColorMap = sColorMap, 
                         aDataObserved = aDataObserved, tStyleDataObs = tStyleDataObs_var, tLimX = tLimX, tLimY = tLimY )
    
    if ( aZ is not None ):
        sTitleText = sTitleText + "(MSE ($e_{MSE} = %.3f$))" % ( np.mean ( aMSE ) )
        CGraCon.Set ( sTitle = sTitleText, sLegend = "$MSE\,(x, y)$" )
        pl.PlotContour ( aX = aGridX, aY = aGridY, aData2D = aMSE, iNumLevel = 6, GraphicConfig = CGraCon, sColorMap = sColorMap, 
                         aDataObserved = aDataObserved, tStyleDataObs = tStyleDataObs_data, tLimX = tLimX, tLimY = tLimY )
    
    return
# *********************************** Schätzung für Mittelwert und Varianz basierend auf Kriging-Modell ************************************  
def DemoKriging ( tDimX = ( 0.0, 6.0, 50 ), tDimY = ( 0.0, 5.0, 40 ), iNumLevel = 20 ):
    CheckAssert ( bBool = ( ( len ( tDimX ) == 3 ) and ( len ( tDimY ) == 3 ) ), sMsg = "Invalid Parameter Shape" )
    # conditioning data
    tCond_x = ( 0.3, 1.9, 1.1, 3.3, 4.7, 2.2, 4.1, 0.7, 2.9, 3.8 )
    tCond_y = ( 1.2, 0.6, 3.2, 4.4, 3.8, 2.3, 3.0, 2.9, 0.6, 2.8 )
    tCond_val = ( 0.47, 0.56, 0.74, 1.47, 1.74, 1.12, 0.98, 1.33, 1.09, 1.33 )

    aDataObserved = np.asarray ( ( tCond_x, tCond_y, tCond_val ), dtype = np.float64 )
    print ( aDataObserved )

    CCovModel = CCovarianceModelGsT ( sModel = "rational", fVar = 0.5, uLenScale = 1.0, fShape = 1.5, fAngles = 0.0, fNugget = 0.0 )
    
    COrdKriging = COrdKrigingGsT ( CCovarianceModel = CCovModel, uDataObserved = aDataObserved, bFitVariogram = True )
    aRField = COrdKriging.Interpolate ( tDimX = tDimX, tDimY = tDimY, iNumLevel = iNumLevel )
    
    ### sollte exakt so aussehen wie InterpolateGrid
    COrdKriging.Predict ( uPos = aDataObserved[ : 2, : ], aDataObserved = aDataObserved[ 2, : ] )
    
    aMSE = COrdKriging.RunCrossValidation ( iNumFolds = 5 )
    print ( ">> Demokriging > MSE of CV: %.2f" % ( np.mean ( aMSE ) ) )

    CGraCon = pl.CGraphicConfig ( sTitle = "Kriging (Image)", sGridAxis = "both" )
    pl.PlotImage ( aData2Dim = aRField[ 0 ], GraphicConfig = CGraCon, sColorMap = "RdYlBu_r", 
                   sInterpolation = "spline36", sOrigin = "lower", tExtent = ( 0, 6, 0, 5), sGridAxis = "both")

    return
# ************************************* Demonstration des Kriging anhand eines Bildes (für die Einleitung ?) *******************************
def DemoKrigingPortrait ( fRatio = 0.4, iScaleFactor = 12 ): # 12
    sPathImageFile = "C:\\DATA\\Bilder\\Andre\\Privat\\Andre_Kindergarten.jpg"
    
    aImageArray = RescaleImageFile ( sPathImageFile = sPathImageFile, iScaleFactor = iScaleFactor, bReturnImagePIL = False )
    aGrayImage = ConvertRGBImageArrayToGray ( aImageArray = aImageArray ) 
    iDimY, iDimX = aGrayImage.shape
    print ( "Anzahl Pixel: %d" % ( aGrayImage.size ) )

    CVario = CVariogramSkG ( aData = aGrayImage, sEstimator = "cressie", iNumLags = 45, fMaxLag = 50, sColorMap = "Grays_r" )
    CVario.Fit ( sModel = "exponential", bUseBounds = False, bShowFit = True ) ## exponential geht gut
    #CVario.CompareCovModel ()
    CCovModel_gst = CVario.CVariogram_skg.to_gstools ()

    aSampleData2D, tCoords, tObservations = SampleFromData2D ( aData2Dim = aGrayImage, fRatio = fRatio, fEmptyValue = 1.0, sColorMap = "Grays_r" )  

    COrdKriging = COrdKrigingGsT ( CCovarianceModel_gst = CCovModel_gst, uDataObserved = ( tCoords[ 0 ], tCoords[ 1 ], tObservations ), 
                                   bFitVariogram = True )
    aRField = COrdKriging.Interpolate ( tDimY = ( 0, iDimX, iDimX ), tDimX = ( 0, iDimY, iDimY ), iNumLevel = None, sColorMap = "Grays_r" )

    pl.PlotImagesNx3 ( tImageArray = ( aGrayImage, aSampleData2D, aRField[ 0 ] ), sTickOption = "none", 
                       tTitle = ( "Original", "Beobachtung (%.0f%%)" % ( fRatio * 100.0 ), "Ergebnis \"Kriging\"" ), sColorMap = "Grays_r" )
    
    pl.PlotImagesNx2 ( tImageArray = ( aGrayImage, aRField[ 0 ] ), sTickOption = "all",
                       tTitle = ( "Original", "Ergebnis \"Kriging\"" ), sColorMap = "Grays_r" )
    
    return

# ****************************** Beispiel für Semivarianz-Schätzer für Kapitel "Theorie: Semivarianz-Schätzung" ****************************
def DemoVariogramEstimators ():
    sPathImageFile = "C:\\DATA\\Bilder\\Andre\\Privat\\Andre_Kindergarten.jpg"
    aImageArray = RescaleImageFile ( sPathImageFile, iScaleFactor = 12, bReturnImagePIL = False ) # 12
    aGrayImage = ConvertRGBImageArrayToGray ( aImageArray = aImageArray ) 
    aSampleData2D, tCoords, tObservations = SampleFromData2D ( aData2Dim = aGrayImage, fRatio = 0.4, fEmptyValue = 1.0, sColorMap = None ) 
    iDimY, iDimX = aGrayImage.shape
    #print ( "Anzahl Pixel: %d" % ( aGrayImage.size ) )

    CRaFi = CRandomFieldGsT ( tDim = ( 100, 100 ), sModel = "gaussian", uLenScale = 10.0, fMean = 0.0 )
    #CRaFi.Show ( sColorMap = "RdYlBu", sTitleText = "Isotropes Gauß'sches Zufallsfeld" )
    aRandomField = CRaFi.aRandomField

    CVario = CVariogramSkG ( aData = aRandomField, sEstimator = "matheron", uMaxLag = 30 )
    CVario.EstimateVariogram ( tParameterEstimator = ( ( "matheron", 30 ), ( "cressie", 30 ) ), sColorMap = "RdYlBu",
                               sImageTitle = "Isotropes Gauß'sches Zufallsfeld" )

    CVario = CVariogramSkG ( aData = aSampleData2D, sEstimator = "matheron", uMaxLag = 40 )
    CVario.EstimateVariogram ( tParameterEstimator = ( ( "matheron", 40 ), ( "cressie", 40 ) ), sColorMap = "Grays_r", 
                               sImageTitle = "Grauwert-Bild" )

    return



