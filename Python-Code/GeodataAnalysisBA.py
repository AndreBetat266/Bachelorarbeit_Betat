# -*- coding: utf-8 -*-
# Version vom 21. Juni 2026

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PlotBA import GetColor, GDictPlotParameter, DrawFrameAxis, PlotScatterwMap, CGraphicConfig
import osmnx as ox
from UtilitiesBA import CheckAssert, CountFrequencyContinousData
from osmnx import plot_graph
from geopandas.geodataframe import GeoDataFrame
from networkx.classes.multidigraph import MultiDiGraph
from scipy.spatial import distance

### Definiert welche Daten importiert werden für Messdaten von VERSCHIEDENEN Sensoren aus der NPY-Datei
GDictDataColumns = {
    "GeoP1"        : ( ( 0, 1, 4 ),    ( "Längengrad (°)", "Breitengrad (°)", "$\mathrm{P}_{10}$", "$\mu g/m^3$" ), "MATTER" ),
    "GeoP2"        : ( ( 0, 1, 5 ),    ( "Längengrad (°)", "Breitengrad (°)", "$\mathrm{P}_{2.5}$", "$\mu g/m^3$" ), "MATTER" ),
    "AzmP1"        : ( ( 2, 3, 4 ),    ( "Östliche Entfernung vom Zentrum (m)", "Nördliche Entfernung vom Zentrum (m)", 
                                         "$\mathrm{P}_{10}$", "$\mu g/m^3$" ), "MATTER" ),
    "AzmP2"        : ( ( 2, 3, 5 ),    ( "Östliche Entfernung vom Zentrum (m)", "Nördliche Entfernung vom Zentrum (m)", 
                                         "$\mathrm{P}_{2.5}$", "$\mu g/m^3$" ), "MATTER" ),
    "AzmP1P2"      : ( ( 2, 3, 4, 5 ), ( "Östliche Entfernung vom Zentrum (m)", "Nördliche Entfernung vom Zentrum (m)", 
                                         "$\mathrm{P}_{10}$", "$\mu g/m^3$", "$\mathrm{P}_{2.5}$", "$\mu g/m^3$" ), "MATTER" ),
    "AzmT"         : ( ( 2, 3, 4 ),    ( "Östliche Entfernung vom Zentrum (m)", "Nördliche Entfernung vom Zentrum (m)", 
                                         "$\\theta$", "$°C$" ), "TEMPERATURE" )
    }


# ******************************** Anzeigen einer Location und Rückgabe des umschreibenden Polygon-Zugs ************************************
###                             sPlaceName = "Munich, Germany" #sPlaceName = "Lueneburg, Germany" #sPlaceName = "Hamburg, Germany"
def ShowBorder ( sPlaceName, sGraphType = "straßen_grob", CProjCRS = None, bShow = False ):
    sGraphType = sGraphType.lower ()
    CheckAssert ( bBool = ( sGraphType in [ "straßen_fein", "straßen_mittel", "straßen_grob", "öpnv_ubahn", "none" ] ), 
                  sMsg = "Invalid Choice for <sGraphType>!" )
    ### residential highway (highway=residential in OpenStreetMap) refers to a street or road designed primarily for 
    ### local access to residential housing, rather than for connecting different settlements or handling through traffic
    DictCustomFilter = {
        "straßen_fein"      : '["highway"~"motorway|motorway_link|primary|secondary|tertiary|residential"]',
        "straßen_mittel"    : '["highway"~"motorway|motorway_link|primary|secondary|tertiary"]',
        "straßen_grob"      : '["highway"~"motorway|motorway_link|primary|secondary"]',
        "öpnv_ubahn"        : '["railway"~"subway"]',
        "park"              : '["leisure"~"park|grass"]',
        }
    sCustomFilter = DictCustomFilter.get ( sGraphType ) # ( "park" )
    
    CGeoDfPlace = ox.geocode_to_gdf ( sPlaceName )
    if ( CProjCRS is not None ):
        CGeoDfPlace = ox.projection.project_gdf ( gdf = CGeoDfPlace, to_crs = CProjCRS, to_latlong = False )
    
    aBoundingBox = np.squeeze ( np.asarray ( CGeoDfPlace.bounds ) )
    CGeoDsGeometry = CGeoDfPlace[ "geometry" ]
    CMultiPolygon = CGeoDsGeometry.loc[ 0 ]
    if ( sGraphType != "none" ):
        CMultiGraph = ox.graph_from_place ( sPlaceName, network_type = "drive", custom_filter = sCustomFilter, simplify = True )
        if ( CProjCRS is not None ):
            CMultiGraph = ox.projection.project_graph ( G = CMultiGraph, to_crs = CProjCRS, to_latlong = False )
    
    if ( bShow == True ):
        CFigure, CAxis = plt.subplots ( nrows = 1, ncols = 1, figsize = ( 16, 10 ) )        
        CGeoDfPlace.plot ( kind = "geo", ax = CAxis, edgecolor = GetColor ( "c12" ), linewidth = 2.0, facecolor = GetColor ( "s2" ), 
                           alpha = 0.9 )
        if ( sGraphType != "none" ):
            ox.plot_graph ( G = CMultiGraph, ax = CAxis, bgcolor = "none", node_size = 0, edge_color = GetColor ( "c12" ), 
                            edge_linewidth = 0.1, edge_alpha = 0.9, show = False, bbox = aBoundingBox ) 
        
        #CGeoDfPlace.plot ( kind = "geo", ax = CAxis, edgecolor = pl.GetColor ( "b12" ), facecolor = pl.GetColor ( "s2" ), alpha = 0.9 )
    
        #CGraphPlace_proj = ox.project_graph ( CGraphPlace )
        #print ( type ( CGraphPlace_proj ) )
        #CGeoDfDrive_proj = ox.graph_to_gdfs ( CGraphPlace_proj, nodes = False )
        #CGeoDfDrive_proj.plot ()
        #print ( type ( CGeoDfDrive_proj ) )
    
        plt.show ()

    if ( sGraphType != "none" ):
        return ( CGeoDfPlace, CMultiPolygon, CMultiGraph )
    else:
        return ( CGeoDfPlace, CMultiPolygon )
# ****************************** Plot einer Karte und einem Netzwerk als Hintergrund zu einem Scatter-Plot *********************************
###                             tStyleGeoDataEdge = ( sGeoDataEdgeColor, fGeoDataEdgeLineWidth )
###                             tMarker = ( X, Y, sMarker, fMarkerSize, sColor )
def PlotScatterwMapAndGraph ( aX, aY, aZ, tMarker, GraphicConfig, tStyle, DfGeoData, CMultiGraph, tStyleGeoDataEdge, 
                              sGeoDataFaceColor = "none", tStyleRectangle = None, fAlpha = 1.0 ):
    CheckAssert ( bBool = ( aX.ndim == 1 and aY.ndim == 1 ), sMsg = "aX and aY must be 1-dimensional!" )
    CheckAssert ( bBool = ( ( aX.shape == aY.shape ) and ( aX.shape == aZ.shape ) ), sMsg = "Dimension mismatch!" )
    CheckAssert ( bBool = ( len ( tStyle ) == 4 ), sMsg = "<tStyle> must be 4-dimensional!" )    
    CheckAssert ( bBool = ( len ( tStyleGeoDataEdge ) == 2 ), sMsg = "<tStyleGeoDataEdge> must be 2-dimensional!" )    
    CheckAssert ( bBool = ( isinstance ( DfGeoData, GeoDataFrame ) ), sMsg = "Wrong Format <DfGeoData>!" )
    CheckAssert ( bBool = ( isinstance ( CMultiGraph, MultiDiGraph ) ), sMsg = "Wrong Format <CMultiGraph>!" )
        
    sColor, sMarker, fMarkerSize, sLabel  = tStyle
    sGeoDataEdgeColor, fGeoDataEdgeLineWidth = tStyleGeoDataEdge

    CFigure, CAxis = plt.subplots ( nrows = 1, ncols = 1, figsize = GraphicConfig.tFigureSize )

    CAxis2 = DfGeoData.plot ( kind = "geo", ax = CAxis, edgecolor = GetColor ( sGeoDataEdgeColor ), 
                              linewidth = fGeoDataEdgeLineWidth, facecolor = GetColor ( sGeoDataFaceColor ), alpha = fAlpha )
    
    if ( tStyleRectangle is not None ):
        CheckAssert ( bBool = ( len ( tStyleRectangle ) == 8 ), sMsg = "Invalid Shape of <tStyleRectangle>!",
                     sExtraInfo = "fLowerLeftX, fLowerLeftY, fWidth, fHeight, fLineWidth, sLineStyle, sEdgeColor, sFaceColor" )
        fLowerLeftX, fLowerLeftY, fWidth, fHeight, fLineWidth, sLineStyle, sEdgeColor, sFaceColor = tStyleRectangle
        CRect = Rectangle ( ( fLowerLeftX, fLowerLeftY ), fWidth, fHeight, linewidth = fLineWidth, linestyle = sLineStyle, 
                              edgecolor = GetColor ( sEdgeColor) , facecolor = GetColor ( sFaceColor ) )

        CAxis.add_patch ( CRect )
    
    tLimX = CAxis2.get_xlim ()
    tLimY = CAxis2.get_ylim ()
    
    plot_graph ( G = CMultiGraph, ax = CAxis, bgcolor = "none", node_size = 0, edge_color = GetColor ( "c12" ), 
                 edge_linewidth = 0.3, edge_alpha = fAlpha, show = False, close = False,
                 bbox = [ tLimX[ 0 ], tLimY[ 0 ], tLimX[ 1 ], tLimY[ 1 ] ] )
                 
    if ( sColor in plt.colormaps () ):
        CPathCollection = CAxis.scatter ( x = aX, y = aY, s = fMarkerSize, marker = sMarker, c = aZ, cmap = sColor, label = sLabel )
    else:
        CPathCollection = CAxis.scatter ( x = aX, y = aY, s = fMarkerSize, marker = sMarker, c = GetColor ( sColor ), cmap = None, label = sLabel )
    
    if ( tMarker is not None ):
        for ik in range ( len ( tMarker ) ):
            fX, fY, sMarker, fMarkerSize, sColor = tMarker[ ik ] 
            CPathCollection = CAxis.scatter ( x = fX, y = fY, s = fMarkerSize, marker = sMarker, color = GetColor ( sColor ) )
                                              #edgecolor = pl.GetColor ( sColor ), facecolor = "none" )
    
    if ( GraphicConfig.sTextLegend ):
        CAxisDivider = make_axes_locatable ( CAxis )
        CAxis2 = CAxisDivider.append_axes ( "right", size = "3%", pad = GraphicConfig.fColorbarPad )
        CColorBar = CFigure.colorbar ( mappable = CPathCollection, cax = CAxis2 )
        
        CColorBar.ax.set_ylabel ( GraphicConfig.sTextLegend, fontname = GDictPlotParameter.get ( "FontName" ), 
                                  fontsize = GDictPlotParameter.get ( "LabelSizeColorbar" ), rotation = -90, verticalalignment = "bottom" )

        for CLabel in CColorBar.ax.get_yticklabels ():
            CLabel.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
            CLabel.set_fontsize ( fontsize = GDictPlotParameter.get ( "TickSizeColorbar" ) )
        
    DrawFrameAxis ( CAxis = CAxis, GraphicConfig = GraphicConfig, iIndex = 0, sGridAxis = GraphicConfig.sGridAxis )
    plt.show ()
    
    return

# ************* Grafische Darstellung der Sensor-Daten für einen bestimmten Tag und eine bestimmte Uhrzeit pm WindowDeltaMinutes ***********
###                     Hintergrund der Umriß und das Netzwerk innerhalb des gewählten Areals
def ShowDataSnapShotwGraph ( aRawData, sDataSelection, sDate, sStartTime = None, iWindowDelta = None, CProjCRS = None, 
                             tStyleRectangle = None, bShowDistribution = True ): 
    ## lat_geo, lon_geo, lat_prj, lon_prj, P1-Messwert, P2-Messwert
    CheckAssert ( bBool = ( aRawData.shape[ 1 ] in [ 5, 6 ] ), sMsg = "Invalid Shape for <aData>!", sExtraInfo = "(%s)" % ( str ( aRawData.shape ) ) ) 
    tMarker = None
    aSelect, tLabel, _ = GDictDataColumns.get ( sDataSelection, ( None, None ) )
    sLabelX, sLabelY, sDescription, sUnit = tLabel
    aData = aRawData[ :, aSelect ]

    DfGeoData, _, CMultiGraph = ShowBorder ( sPlaceName = "Munich, Germany", sGraphType = "straßen_mittel", CProjCRS = CProjCRS, bShow = False )

    aX = aData[ :, 1 ]
    print ( ">> X-Values > Shape: %s, Min: %.1f, Max: %.1f" % ( str ( aX.shape ), np.amin ( aX ), np.amax ( aX ) ) )
    aY = aData[ :, 0 ]
    print ( ">> Y-Values > Shape: %s, Min: %.1f, Max: %.1f" % ( str ( aY.shape ), np.amin ( aY ), np.amax ( aY ) ) )
    aZ = aData[ :, 2 ] 
    print ( ">> Z-values > Shape: %s, Min: %.1f, Max: %.1f" % ( str ( aZ.shape ), np.amin ( aZ ), np.amax ( aZ ) ) )

    if ( ( sStartTime is not None ) and ( iWindowDelta is not None ) ):
        sTitleText = "%s Messpunkte am %s um %s$\,\\pm %d\,$min (n = %d)" % ( sDescription, sDate, sStartTime, iWindowDelta, aRawData.shape[ 0 ] )
    else:
        sTitleText = "%s Messpunkte im %s (n = %d)" % ( sDescription, sDate, aRawData.shape[ 0 ] )
    #sTitleText = "Stadtgrenze, Messpunkte und Englischer Garten"
    CGraCon = CGraphicConfig ( sTitle = sTitleText, sLabelX = sLabelX )#, sLegend = GDictSensorColumnDescription.get ( "PMS5003" )[ iIndex ], 
                                  #fColorbarPad = -1.4 )

    if ( CProjCRS is not None ):
        tMarker = [ ( 0.0, 0.0, "X", 120.0, "s11" ) ]
    
    PlotScatterwMapAndGraph ( aX = aX, aY = aY, aZ = aZ, tMarker = tMarker, GraphicConfig = CGraCon, 
                              tStyle = ( "g11", "o", 30.0, ""), DfGeoData = DfGeoData, CMultiGraph = CMultiGraph, 
                              tStyleGeoDataEdge = ( "s14", 2.0 ), sGeoDataFaceColor = "s1", tStyleRectangle = tStyleRectangle, 
                              fAlpha = 0.9 )
    
    ##geändert für darasetllung Englsicher Garten
    """
    PlotScatterwMapAndGraph ( aX = aX, aY = aY, aZ = aZ, tMarker = tMarker, GraphicConfig = CGraCon, 
                              tStyle = ( "s11", "x", 15.0, ""), DfGeoData = DfGeoData, CMultiGraph = CMultiGraph, 
                              tStyleGeoDataEdge = ( "s14", 2.0 ), sGeoDataFaceColor = "s1", tStyleRectangle = tStyleRectangle, 
                              fAlpha = 0.9 )
    """
    if ( bShowDistribution == True ):
        ShowDistributionDistances ( aData = aData, sDescription = sDescription, sUnit = sUnit )
    
    return
# ************* Grafische Darstellung der Sensor-Daten für einen bestimmten Tag und eine bestimmte Uhrzeit pm WindowDeltaMinutes ***********
#                   Hintergrund der Umriß des gewählten Areals
def ShowDataSnapshotwBorder ( aRawData, sDataSelection, sDate, sStartTime = None, iWindowDelta = None, CProjCRS = None, tStyleRectangle = None, 
                              ListAnnotation = None, bShowDistribution = True ):
    ## lat_geo, lon_geo, lat_prj, lon_prj, P1-Messwert, P2-Messwert
    ## lat_geo, lon_geo, lat_prj, lon_prj, Temperatur-Messwert
    CheckAssert ( bBool = ( aRawData.shape[ 1 ] in [ 5, 6 ] ), sMsg = "Invalid Shape for <aData>!", sExtraInfo = "(%s)" % ( str ( aRawData.shape ) ) ) 
    tMarker = None
    aSelect, tLabel, _ = GDictDataColumns.get ( sDataSelection, ( None, None ) )
    sLabelX, sLabelY, sDescription, sUnit = tLabel
    aData = aRawData[ :, aSelect ]
    
    CGeoDfPlace, _ = ShowBorder ( sPlaceName = "Munich, Germany", sGraphType = "none", CProjCRS = CProjCRS )

    aX = aData[ :, 1 ]
    print ( ">> X-Values > Shape: %s, Min: %.1f, Max: %.1f" % ( str ( aX.shape ), np.amin ( aX ), np.amax ( aX ) ) )
    aY = aData[ :, 0 ]
    print ( ">> Y-Values > Shape: %s, Min: %.1f, Max: %.1f" % ( str ( aY.shape ), np.amin ( aY ), np.amax ( aY ) ) )
    aZ = aData[ :, 2 ] 
    print ( ">> Z-Values > Shape: %s, Min: %.1f, Max: %.1f" % ( str ( aZ.shape ), np.amin ( aZ ), np.amax ( aZ ) ) )
    
    iIndexMaxZ = np.argmax ( aZ )
    print ( ">> Maximum of %.1f at (%.2f, %.2f)" % ( aZ[ iIndexMaxZ], aX[ iIndexMaxZ ], aY[ iIndexMaxZ ] ) )
    
    if ( ( sStartTime is not  None ) and ( iWindowDelta is not None ) ):
        sTitleText = "Messungen %s am %s um %s$\,\\pm %d\,$min (n = %d)" % ( sDescription, sDate, sStartTime, iWindowDelta, aRawData.shape[ 0 ] )
    else:
        sTitleText = "Messungen %s im %s (n = %d)" % ( sDescription, sDate, aRawData.shape[ 0 ] )
        
    CGraCon = CGraphicConfig ( sTitle = sTitleText, sLabelX = sLabelX, sLabelY = sLabelY, sLegend = sDescription + " " + sUnit, fPosVariable = 0.5 )

    if ( CProjCRS is not None ):
        tMarker = [ ( 0.0, 0.0, "X", 180.0, "s11" ) ] 
     
    PlotScatterwMap ( aX = aX, aY = aY, aZ = aZ, tMarker = tMarker, GraphicConfig = CGraCon, tStyle = ( "RdYlBu_r", "o", 80, "" ), 
                         DfGeoData = CGeoDfPlace, tStyleGeoDataEdge = ( "s14", 2 ), sGeoDataFaceColor = "s1", 
                         tStyleRectangle = tStyleRectangle, fAlpha = 0.9 )
    
    if ( bShowDistribution == True ):
        ShowDistributionDistances ( aData = aData, sDescription = sDescription, sUnit = sUnit )
    
    return

# ******************************************* Analyse der Verteilung der Distanzen *********************************************************
def ShowDistributionDistances ( aData, sDescription, sUnit ):
    CheckAssert ( bBool = ( aData.shape[ 1 ] >= 3 ), sMsg = "Invalid Shape for <aData>!", sExtraInfo = "%s" % ( str ( aData.shape ) ) )
    aDist = np.zeros ( shape = ( aData.shape[ 0 ], aData.shape[ 0 ] ), dtype = np.float32 )

    for iZeile in range ( aDist.shape[ 0 ] ):
        for iSpalte in range ( aDist.shape[ 1 ] ): 
            aDist[ iZeile ][ iSpalte ] = distance.euclidean ( ( aData[ iZeile, 0 ], aData[ iZeile, 1 ] ), 
                                                             ( aData[ iSpalte, 0 ], aData[ iSpalte, 1 ] ) )
            
    aDist_diff = aDist[ np.triu_indices ( aDist.shape[ 0 ], 1 ) ]

    sTitleText = "Verteilung der Euklidischen Distanzen ($N=%d$)" % ( aDist_diff.shape[ 0 ] )
    CGraCon = CGraphicConfig ( sTitle = sTitleText, sLabelX = "Euklidische Distanz $R$ (m)", 
                                  sLabelY = "Relative Häufigkeit" )
    CountFrequencyContinousData ( aData = aDist_diff, iNumBins = 20, sReturnType = "relative", bCenterEdges = True, 
                                  sColor = "g10", GraphicConfig = CGraCon, bInfo = False )
    
    sTitleText = "Verteilung der %s-Messwerte ($N=%d$)" % ( sDescription, aData.shape[ 0 ] )
    sTextLabelX = "%s %s" % ( sDescription, sUnit )
    CGraCon.Set ( sTitle = sTitleText, sLabelX = sTextLabelX )
    CountFrequencyContinousData ( aData[ :, 2 ], iNumBins = 30, sReturnType = "relative", bCenterEdges = True, sColor = "b10", 
                                  GraphicConfig = CGraCon, bInfo = False )
    
    ### nicht sooo hilffreich
    #CGraCon.Set ( sTitle = "Visualisierung der relativen Abstände", sLabelX = "Nr. Sensor", sLabelY = "Nr. Sensor", sLegend = "Euklidische Distanz" )
    #pl.PlotImage ( aData2Dim = aDist, GraphicConfig = CGraCon, sColorMap = "Greys" )  
    
    return
# ******************************************* Analyse der Verteilung der Messwerte *********************************************************
def ShowDistributionMeasurements ( aData, sTitleStartText, sDescription, sUnit, iNumBins = 40 ):
    CheckAssert ( bBool = ( aData.ndim == 1 ), sMsg = "Invalid Shape for <aData>!" )

    sTitleText = sTitleStartText + ": Verteilung der %s-Messwerte ($N=%d$)" % ( sDescription, aData.shape[ 0 ] )
    CGraCon = CGraphicConfig ( sTitle = sTitleText, sLabelX = sDescription + " (" +  sUnit + ")", 
                                  sLabelY = "Relative Häufigkeit", sGridAxis = "y" )
    CountFrequencyContinousData ( aData = aData, iNumBins = iNumBins, sReturnType = "relative", bCenterEdges = True, 
                                  sColor = "b10", GraphicConfig = CGraCon, bInfo = False )
    
    return
# ************************ Umriss von München mit Englischem Garten und Hauptverkehrsstraßen sowie ggf Wasserflächen ***********************
def ShowMunichwBorder ( bAddWater ):
    # 1) Administrative Grenze von München laden
    CGeoDfMunichBboundary = ox.geocode_to_gdf ( "München, Germany" )

    # 2) Englischen Garten als Fläche laden
    CGeoDfEnglischerGarten = ox.geocode_to_gdf ( "Englischer Garten, München, Germany" )

    # 3) Hauptverkehrsstraßen innerhalb der Münchner Stadtgrenze laden
    sMajorRoadsFilter = '["highway"~"motorway|primary|secondary|tertiary"]'

    CGraph = ox.graph_from_polygon ( CGeoDfMunichBboundary.geometry.iloc[ 0 ],
                                     network_type = "drive",
                                     custom_filter = sMajorRoadsFilter, simplify = True, )

    CGeoDfEdges = ox.graph_to_gdfs ( CGraph, nodes = False )

    # 4) Wasserflächen (Flüsse, Seen) innerhalb der Stadtgrenze
    if ( bAddWater == True ):
        DictWaterTags = { "natural": "water", "waterway" : "riverbank" }
        try:
            CGraphWater = ox.features_from_polygon ( CGeoDfMunichBboundary.geometry.iloc[ 0 ], DictWaterTags )
        except Exception as e:
            print ( "Keine Wasserflächen gefunden:", e )
            CGraphWater = None
            
    # 5) Marienplatz als Punkt geocodieren
    CPointMarienplatz = ox.geocode ( "Marienplatz, München, Germany" )  # (lat, lon)
    fLat, fLon = CPointMarienplatz

    # --- Plot zusammenbauen ---
    CFigure, CAxis = plt.subplots ( figsize = ( 12, 12 ) )

    # a) Stadtgrenze (nur Umriss)
    CGeoDfMunichBboundary.boundary.plot ( ax = CAxis, color = GetColor ( "s14" ), linewidth = 2.0, zorder = 1 )

    # Wasserflächen hellblau
    if ( bAddWater == True ):
        if ( ( CGraphWater is not None ) and ( not CGraphWater.empty ) ):
            CGraphWater.plot ( ax = CAxis, color = GetColor ( "b4" ), zorder = 2 )

    # b) Englischer Garten grün ausgefüllt
    CGeoDfEnglischerGarten.plot ( ax = CAxis, color = GetColor ( "g12" ), alpha = 0.9, zorder = 3 )

    # c) Hauptverkehrsstraßen dunkelbraun
    CGeoDfEdges.plot ( ax = CAxis, color = GetColor ( "c12" ), linewidth = 0.5, zorder = 4 )

    # d) Marienplatz als graues Kreuz
    CAxis.scatter ( fLon, fLat, marker = "X", color = GetColor ( "s11" ), s = 180, linewidths = 2.5, zorder = 6 )

    CAxis.set_title ( "München: Stadtgrenze, Englischer Garten, Hauptstraßen", 
                       fontname = GDictPlotParameter.get ( "FontName" ), fontsize = 20 )
    CAxis.set_axis_off ()

    plt.tight_layout ()

    plt.show ()
    
    return
    
