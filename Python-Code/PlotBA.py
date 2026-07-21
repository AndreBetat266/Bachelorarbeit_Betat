# -*- coding: utf-8 -*-
# Version vom 21. Juli 2026

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from UtilitiesBA import CheckAssert
from geopandas.geodataframe import GeoDataFrame


# ~~~~~~~~~~~~~~~~~~~ Klasse und übergeordnete Funktionen, die allgemeinn zur Anwendung kommen bei der Erstellung von Grafiken ~~~~~~~~~~~~~

# ******************************************* Dictionary der gängistsen Parameter **********************************************************
GDictPlotParameter = {
    "FigureSizeStackVert"       : ( 12, 13 ),
    "FigureSizeStackVert2x2"    : ( 18, 13 ),
    "FigureSize"                : ( 12, 7.42 ),
    "FigureSizeStack2Horz"      : ( 13, 7.42 ),### geändert für Anisotrope Kapitel auf ( 12, 5 )
    "FigureSizeStack3Horz"      : ( 18, 7.42 ),
    "FontName"                  : "Century Gothic",
    "TitleSize"                 : 18,
    "TitleSizeL"                : 20,
    "TitleSize3HorzStackImages" : 14,
    "TitleSize2HorzStackImages" : 18,
    "LabelSize"                 : 16,
    "LabelSizeL"                : 18,
    "LabelSizeColorbar"         : 14, 
    "TickSize"                  : 14,
    "TickSizeL"                 : 16,
    "TickSizeColorbar"          : 12, 
    "TickSize3HorzStackImages"  : 10,
    "TickSize2HorzStackImages"  : 14,
    "Annotation"                : 8,
    "LegendSize"                : 12,
    "LegendPosition"            : "best",
    "InfoBoxTextSize"           : 12,
    "InfoBoxAlpha"              : 0.5,
    "InfoBoxFaceColor"          : "azure",
    "InfoBoxStyle"              : "round"
    }
# ************************************************************* Klasse CInfoBox ************************************************************
class CInfoBox ( object ):
    fAlpha : float = GDictPlotParameter.get ( "InfoBoxAlpha" )
    fBoxPosX : float = None
    fBoxPosY : float = None
    iBoxFontSize : int = GDictPlotParameter.get ( "InfoBoxTextSize" )
    sBoxStyle : str = GDictPlotParameter.get ( "InfoBoxStyle" )
    sFaceColor : str = GDictPlotParameter.get ( "InfoBoxFaceColor" )
    sText : str = ""
    #DictBoxProps : dict = dict ( boxstyle = GDictPlotParameter.get ( "InfoBoxStyle" ), facecolor = GDictPlotParameter.get ( "InfoBoxFaceColor" ),
    #                             alpha = GDictPlotParameter.get ( "InfoBoaxAlpha" ) )
    
    def __init__ ( self, sText = "", fBoxPosX = None, fBoxPosY = None, iBoxFontSize = None, sFaceColor = "", fAlpha = None ):
        if ( fAlpha is not None ):
            self.fAlpha = fAlpha
        if ( fBoxPosX is not None ):
            self.fBoxPosX = fBoxPosX
        if ( fBoxPosY is not None ):
            self.fBoxPosY = fBoxPosY
        if ( iBoxFontSize is not None ):
            self.iBoxFontSize = iBoxFontSize
        if ( sFaceColor ):
            self.sFaceColor = sFaceColor
        if ( sText ):
            self.sText = sText
            
        self.DictBoxProps = dict ( boxstyle = self.sBoxStyle, facecolor = self.sFaceColor, alpha = self.fAlpha )
        
        return
# ************************************************************* Klasse CLine *************************************************************
class CLine ( object ):
    sLineColor : str = ""
    fLinePos : float = None
    fLineWidth : float = 3.0
    sLineStyle : str = "--"
    sLineLabel : str = "_nolegend_"
    
    def __init__ ( self, sLineColor = "", fLinePos = None, fLineWidth = None, sLineStyle = "", sLineLabel = "" ):
        if ( sLineColor ):
            self.sLineColor = sLineColor
        if ( fLinePos is not None ):
            self.fLinePos = fLinePos
        if ( fLineWidth is not None ):
            self.fLineWidth = fLineWidth
        if ( sLineStyle ):
            self.sLineStyle = sLineStyle
        if ( sLineLabel ):
            self.sLineLabel = sLineLabel
        
        return
# ****************************** Klasse, die alle nicht direkt dem Plot zuzuordnenden Elemente verwaltet **********************************
class CGraphicConfig ( object ):
## geändert 15.03.2026: alle Text-Info nun vierfach, sowie viermal horizontale und vertikale Linien 
    sTextTitle : str = ""
    sTextTitle2 : str = ""
    sTextTitle3 : str = ""
    sTextTitle4 : str = ""
    sTextLabelX : str = ""
    sTextLabelX2 : str = ""
    sTextLabelX3 : str = ""
    sTextLabelX4 : str = ""
    sTextLabelY : str = ""
    sTextLabelY2 : str = ""
    sTextLabelY3 : str = ""
    sTextLabelY4 : str = ""
    sTextLabelZ : str = ""
    sTextLegend : str = ""
    sTextLegend2 : str = ""
    sTextLegend3 : str = ""
    sTextLegend4 : str = ""
    sLegendPosition : str = GDictPlotParameter.get ( "LegendPosition" ) 
    sGridAxis : str = "both"
    sStepPlotWhere : str = "none"
    sAnnotationVertAlign : str = "bottom"
    sAnnotationHorzAlign : str = "left"
    sMarkerSingleLabel : str = ""
    fPosVariable = 0.2
    
    VLine1 : CLine = CLine ( sLineColor = "r12" )
    VLine2 : CLine = CLine ( sLineColor = "r12" )
    VLine3 : CLine = CLine ( sLineColor = "r12" )
    VLine4 : CLine = CLine ( sLineColor = "r12" )
    HLine1 : CLine = CLine ( sLineColor = "g12" )
    HLine2 : CLine = CLine ( sLineColor = "g12" )
    HLine3 : CLine = CLine ( sLineColor = "g12" )
    HLine4 : CLine = CLine ( sLineColor = "g12" )
    
    InfoBox : CInfoBox = None
    tFigureSize : tuple = GDictPlotParameter.get ( "FigureSize" )
    CFigure = None
    bHideTicksX = False
    bHideTicksY = False
    bShowPlot = True
        
    def __init__ ( self, sTitle = "", sTitle2 = "", sTitle3 = "", sTitle4 = "", 
                   sLabelX = "", sLabelX2 = "", sLabelX3 = "", sLabelX4 = "",
                   sLabelY = "", sLabelY2 = "", sLabelY3 = "", sLabelY4 = "", sLabelZ = "", 
                   sLegend = "", sLegend2 = "", sLegend3 = "", sLegend4 = "", sLegendPosition = "", sGridAxis = "", sStepPlotWhere = "",
                   sAnnotationVertAlign = "", sAnnotationHorzAlign = "", sMarkerSingleLabel = "", fPosVariable = None, VLine1 = None, HLine1 = None, 
                   VLine2 = None, HLine2 = None, VLine3 = None, HLine3 = None, VLine4 = None, HLine4 = None, InfoBox = None , tFigureSize = None, 
                   CFigure = None, bHideTicksX = None, bHideTicksY = None, bShowPlot = None ):
        
        if ( sTitle ): 
            self.sTextTitle = sTitle
        if ( sTitle2 ):
            self.sTextTitle2 = sTitle2
        if ( sTitle3 ):
            self.sTextTitle3 = sTitle3
        if ( sTitle4 ):
            self.sTextTitle4 = sTitle4
                
        if ( sLabelX ):
            self.sTextLabelX = sLabelX
        if ( sLabelX2 ):
            self.sTextLabelX2 = sLabelX2
        if ( sLabelX3 ):
            self.sTextLabelX3 = sLabelX3    
        if ( sLabelX4 ):
            self.sTextLabelX4 = sLabelX4    
            
        if ( sLabelY ):
            self.sTextLabelY = sLabelY
        if ( sLabelY2 ):
            self.sTextLabelY2 = sLabelY2
        if ( sLabelY3 ):
            self.sTextLabelY3 = sLabelY3
        if ( sLabelY4 ):
            self.sTextLabelY4 = sLabelY4
                
        if ( sLabelZ ):
            self.sTextLabelZ = sLabelZ    
        if ( sLegend ):
            self.sTextLegend = sLegend
        if ( sLegend2 ):
            self.sTextLegend2 = sLegend2
        if ( sLegend3 ):
            self.sTextLegend3 = sLegend3
        if ( sLegend4 ):
            self.sTextLegend4 = sLegend4
                
        if ( sLegendPosition ):
            self.sLegendPosition = sLegendPosition
        if ( sGridAxis ):
            self.sGridAxis = sGridAxis
        if ( sStepPlotWhere ):
            self.sStepPlotWhere = sStepPlotWhere
        if ( sAnnotationVertAlign ):
            self.sAnnotationVertAlign = sAnnotationVertAlign
        if ( sAnnotationHorzAlign ):
            self.sAnnotationHorzAlign = sAnnotationHorzAlign
        if ( sMarkerSingleLabel ):
            self.sMarkerSingleLabel = sMarkerSingleLabel
            
        if ( fPosVariable is not None ):
            self.fPosVariable = fPosVariable
            
        if ( VLine1 is not None ):
            self.VLine1 = VLine1
        if ( HLine1 is not None ):
            self.HLine1 = HLine1
        if ( VLine2 is not None ):
            self.VLine2 = VLine2
        if ( HLine2 is not None ):
            self.HLine2 = HLine2
        if ( VLine3 is not None ):
            self.VLine3 = VLine3
        if ( HLine3 is not None ):
            self.HLine3 = HLine3
        if ( VLine4 is not None ):
            self.VLine4 = VLine4
        if ( HLine4 is not None ):
            self.HLine4 = HLine4    
            
        if ( InfoBox is not None ):
            self.InfoBox = InfoBox
        if ( tFigureSize is not None ):
            self.tFigureSize = tFigureSize 
        if ( CFigure is not None ):
            self.CFigure = CFigure
        if ( bHideTicksX is not None ):
            self.bHideTicksX = bHideTicksX
        if ( bHideTicksY is not None ):
            self.bHideTicksY = bHideTicksY
        if ( bShowPlot is not None ):
            self.bShowPlot = bShowPlot
            
        return
    
    def Set ( self, sTitle = "", sTitle2 = "", sTitle3 = "", sTitle4 = "",
              sLabelX = "", sLabelX2 = "", sLabelX3 = "", sLabelX4 = "",
              sLabelY = "", sLabelY2 = "", sLabelY3 = "", sLabelY4 = "", sLabelZ = "", 
              sLegend = "", sLegend2 = "", sLegend3 = "", sLegend4 = "", sLegendPosition = "", sGridAxis = "", sStepPlotWhere = "",
              sAnnotationVertAlign = "", sAnnotationHorzAlign = "", sMarkerSingleLabel = "",  fPosVariable = None, VLine1 = None, HLine1 = None, 
              VLine2 = None, HLine2 = None, VLine3 = None, HLine3 = None, VLine4 = None, HLine4 = None, InfoBox = None, tFigureSize = None, 
              CFigure = None, bHideTicksX = None, bHideTicksY = None, bShowPlot = None ):
        
        self.__init__ ( sTitle = sTitle, sTitle2 = sTitle2, sTitle3 = sTitle3, sTitle4 = sTitle4,
                        sLabelX = sLabelX, sLabelX2 = sLabelX2, sLabelX3 = sLabelX3, sLabelX4 = sLabelX4,
                        sLabelY = sLabelY, sLabelY2 = sLabelY2, sLabelY3 = sLabelY3, sLabelY4 = sLabelY4, sLabelZ = sLabelZ, 
                        sLegendPosition = sLegendPosition, sLegend = sLegend, sLegend2 = sLegend2, sLegend3 = sLegend3, sLegend4 = sLegend4,
                        sGridAxis = sGridAxis, sStepPlotWhere = sStepPlotWhere, sAnnotationVertAlign = sAnnotationVertAlign, 
                        sAnnotationHorzAlign = sAnnotationHorzAlign, sMarkerSingleLabel = sMarkerSingleLabel, fPosVariable = fPosVariable, 
                        VLine1 = VLine1, HLine1 = HLine1, VLine2 = VLine2, HLine2 = HLine2, VLine3 = VLine3, HLine3 = HLine3, 
                        VLine4 = VLine4, HLine4 = HLine4, InfoBox = InfoBox, tFigureSize = tFigureSize, CFigure = CFigure, 
                        bHideTicksX = bHideTicksX, bHideTicksY = bHideTicksY, bShowPlot = bShowPlot )
        
        return
    
    def GetTextLegends ( self ):
        ListLegends = list ()
        ListLegends.append ( self.sTextLegend )
        ListLegends.append ( self.sTextLegend2 )
        ListLegends.append ( self.sTextLegend3 )
        ListLegends.append ( self.sTextLegend4 )
        
        return ( ListLegends )
    
    def GetAllTextInfo ( self ):
        return ( ( self.sTextTitle, self.sTextTitle2, self.sTextTitle3, self.sTextTitle4 ), 
                 ( self.sTextLabelX, self.sTextLabelX2, self.sTextLabelX3, self.sTextLabelX4 ), 
                 ( self.sTextLabelY, self.sTextLabelY2, self.sTextLabelY3, self.sTextLabelY4 ), 
                   self.GetTextLegends () )
    
    def GetAllAttributes ( self, iIndex ):
        CheckAssert ( bBool = ( ( iIndex in [ 0, 1, 2, 3 ] ) ), sMsg = "Invalid Format <iIndex>!" )
        ListHLines, ListVLines = self.GetAllLineInfo ()
        
        CHLine = ListHLines[ iIndex ]
        CVLine = ListVLines[ iIndex ]
    
        if ( iIndex == 0 ):
            tTupel = ( self.sTextTitle, self.sTextLabelX, self.sTextLabelY, self.sTextLegend, CHLine, CVLine )
        elif ( iIndex == 1 ):
            tTupel = ( self.sTextTitle2, self.sTextLabelX2, self.sTextLabelY2, self.sTextLegend2, CHLine, CVLine )
        elif ( iIndex == 2 ):
            tTupel = ( self.sTextTitle3, self.sTextLabelX3, self.sTextLabelY3, self.sTextLegend3, CHLine, CVLine )  
        elif ( iIndex == 3 ):
            tTupel = ( self.sTextTitle4, self.sTextLabelX4, self.sTextLabelY4, self.sTextLegend4, CHLine, CVLine )
                 
        return ( tTupel )
    
    def GetAllLineInfo ( self ):
        tHLines = ( self.HLine1, self.HLine2, self.HLine3, self.HLine4 )
        tVLines = ( self.VLine1, self.VLine2, self.VLine3, self.VLine4 )
        ListHLines = list ()
        ListVLines = list ()
        for ik in range ( len ( tHLines ) ):
            if ( tHLines[ ik ].fLinePos is not None ):
                ListHLines.append ( tHLines[ ik ] )
            else:
                ListHLines.append ( None )
            if ( tVLines[ ik ].fLinePos is not None ):
                ListVLines.append ( tVLines[ ik ] )
            else:
                ListVLines.append ( None )
                
        return ( ListHLines, ListVLines )
    
    def CheckAllLineLabel ( self ):
        bCheckAnyLabel = False
        tLines = ( self.HLine1, self.HLine2, self.HLine3, self.HLine4, self.VLine1, self.VLine2, self.VLine3, self.VLine4 )
        for ik in range ( len ( tLines ) ):
            if ( tLines[ ik ].sLineLabel != "_nolegend_" ):
                bCheckAnyLabel = True
                break
        
        return ( bCheckAnyLabel )

# ************************************************* DrawFrame Funktion *********************************************************************
def DrawFrame ( GraphicConfig, bShowLegend = True ):
    #CFigure = plt.figure ( figsize = CGraphicConfig.tFigureSize ) # keine gute Idee!
    plt.title ( label = GraphicConfig.sTextTitle, fontname = GDictPlotParameter.get ( "FontName" ), fontsize = GDictPlotParameter.get ( "TitleSize" ) )
    plt.xlabel ( xlabel = GraphicConfig.sTextLabelX, fontname = GDictPlotParameter.get ( "FontName" ), fontsize = GDictPlotParameter.get ( "LabelSize" ) )
    plt.ylabel ( ylabel = GraphicConfig.sTextLabelY, fontname = GDictPlotParameter.get ( "FontName" ), fontsize = GDictPlotParameter.get ( "LabelSize" ) )
    plt.xticks ( fontname = GDictPlotParameter.get ( "FontName" ), fontsize = GDictPlotParameter.get ( "TickSize" )  )
    plt.yticks ( fontname = GDictPlotParameter.get ( "FontName" ), fontsize = GDictPlotParameter.get ( "TickSize" ) )
    
    if ( GraphicConfig.VLine1.fLinePos is not None ):
        plt.axvline ( x = GraphicConfig.VLine1.fLinePos, label = GraphicConfig.VLine1.sLineLabel, 
                      color = GetColor ( GraphicConfig.VLine1.sLineColor ), ls = GraphicConfig.VLine1.sLineStyle, 
                      lw = GraphicConfig.VLine1.fLineWidth )
    if ( GraphicConfig.VLine2.fLinePos is not None ):
        plt.axvline ( x = GraphicConfig.VLine2.fLinePos, label = GraphicConfig.VLine2.sLineLabel, 
                      color = GetColor ( GraphicConfig.VLine2.sLineColor ), ls = GraphicConfig.VLine2.sLineStyle, 
                      lw = GraphicConfig.VLine2.fLineWidth )
    if ( GraphicConfig.VLine3.fLinePos is not None ):
        plt.axvline ( x = GraphicConfig.VLine3.fLinePos, label = GraphicConfig.VLine3.sLineLabel, 
                      color = GetColor ( GraphicConfig.VLine3.sLineColor ), ls = GraphicConfig.VLine3.sLineStyle, 
                      lw = GraphicConfig.VLine3.fLineWidth )
    if ( GraphicConfig.HLine1.fLinePos is not None ):
        plt.axhline ( y = GraphicConfig.HLine1.fLinePos, label = GraphicConfig.HLine1.sLineLabel, 
                      color = GetColor ( GraphicConfig.HLine1.sLineColor ), ls = GraphicConfig.HLine1.sLineStyle, 
                      lw = GraphicConfig.HLine1.fLineWidth )
    if ( GraphicConfig.HLine2.fLinePos is not None ):
        plt.axhline ( y = GraphicConfig.HLine2.fLinePos, label = GraphicConfig.HLine2.sLineLabel, 
                      color = GetColor ( GraphicConfig.HLine2.sLineColor ), ls = GraphicConfig.HLine2.sLineStyle, 
                      lw = GraphicConfig.HLine2.fLineWidth )
    if ( GraphicConfig.HLine3.fLinePos is not None ):
        plt.axhline ( y = GraphicConfig.HLine3.fLinePos, label = GraphicConfig.HLine3.sLineLabel, 
                      color = GetColor ( GraphicConfig.HLine3.sLineColor ), ls = GraphicConfig.HLine3.sLineStyle, 
                      lw = GraphicConfig.HLine3.fLineWidth )
            
    if ( GraphicConfig.InfoBox is not None ):
        plt.text ( GraphicConfig.InfoBox.fBoxPosX, GraphicConfig.InfoBox.fBoxPosY, GraphicConfig.InfoBox.sText, 
                   #transform = GraphicConfig.CFigure.transFigure, 
                   fontname = GDictPlotParameter.get ( "FontName" ), fontsize = GraphicConfig.InfoBox.iBoxFontSize, 
                   verticalalignment = "bottom", bbox = GraphicConfig.InfoBox.DictBoxProps )
    
    if ( ( bShowLegend == True ) or ( GraphicConfig.CheckAllLineLabel () == True ) ):
        plt.legend ( prop = { "family": GDictPlotParameter.get ( "FontName" ), "size": GDictPlotParameter.get ( "LegendSize" ) }, 
                     loc = GraphicConfig.sLegendPosition )
    
    if ( GraphicConfig.sGridAxis in [ "x", "y", "both" ] ):
        plt.grid ( visible = True, axis = GraphicConfig.sGridAxis )
    else:
        plt.grid ( visible = False )
    
    return

def DrawFrameAxis ( CAxis, GraphicConfig, iIndex, sGridAxis = "none", tHLines = None, tVLines = None, tSizes = None ):
    if ( tSizes is not None ):
        iTitleSize, iLabelSize, iTickSize = tSizes
    else:
        iTitleSize = GDictPlotParameter.get ( "TitleSize" )
        iLabelSize = GDictPlotParameter.get ( "LabelSize" )
        iTickSize = GDictPlotParameter.get ( "TickSize" )
        
    sTitle, sLabelX, sLabelY, _, CHLine, CVLine = GraphicConfig.GetAllAttributes ( iIndex )    
    CAxis.set_title ( label = sTitle, fontname = GDictPlotParameter.get ( "FontName" ), fontsize = iTitleSize )
    CAxis.set_xlabel ( xlabel = sLabelX, fontname = GDictPlotParameter.get ( "FontName" ), fontsize = iLabelSize )
    CAxis.set_ylabel ( ylabel = sLabelY, fontname = GDictPlotParameter.get ( "FontName" ), fontsize = iLabelSize )
    for CLabel in CAxis.get_xticklabels ():
        CLabel.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
        CLabel.set_fontsize ( fontsize = iTickSize )
    for CLabel in CAxis.get_yticklabels ():
        CLabel.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
        CLabel.set_fontsize ( fontsize = iTickSize )
         
    if ( CHLine is not None ):
        CAxis.axhline ( y = CHLine.fLinePos, label = CHLine.sLineLabel, color = GetColor ( CHLine.sLineColor ),
                        linestyle = CHLine.sLineStyle, linewidth = CHLine.fLineWidth )
        
    if ( ( tHLines is not None ) and ( isinstance ( tHLines, ( tuple, list, np.ndarray ) ) ) ):
        for CHLine in tHLines:
            CAxis.axhline ( y = CHLine.fLinePos, label = CHLine.sLineLabel, color = GetColor ( CHLine.sLineColor ),
                            linestyle = CHLine.sLineStyle, linewidth = CHLine.fLineWidth )
            
    if ( CVLine is not None ):
        CAxis.axvline ( x = CVLine.fLinePos, label = CVLine.sLineLabel, color = GetColor ( CVLine.sLineColor ),
                        linestyle = CVLine.sLineStyle, linewidth = CVLine.fLineWidth )
        
    if ( ( tVLines is not None ) and ( isinstance ( tVLines, ( tuple, list, np.ndarray ) ) ) ):
        for CVLine in tVLines:
            CAxis.axvline ( x = CVLine.fLinePos, label = CVLine.sLineLabel, color = GetColor ( CVLine.sLineColor ),
                            linestyle = CVLine.sLineStyle, linewidth = CVLine.fLineWidth )
    
    if ( GraphicConfig.InfoBox is not None ):
        CAxis.text ( GraphicConfig.InfoBox.fBoxPosX, GraphicConfig.InfoBox.fBoxPosY, GraphicConfig.InfoBox.sText, 
                   #transform = GraphicConfig.CFigure.transFigure, 
                   fontname = GDictPlotParameter.get ( "FontName" ), fontsize = GraphicConfig.InfoBox.iBoxFontSize, 
                   verticalalignment = "bottom", bbox = GraphicConfig.InfoBox.DictBoxProps )
        
    if ( sGridAxis in [ "x", "y", "both" ] ):
        CAxis.grid ( visible = True, axis = sGridAxis )
    else:
        CAxis.grid ( visible = False )
    
    return
# *********************************** Funktionen, die die übergebenen Style Tupel auf Korrektheit prüfen ***********************************
def StyleCheck6 ( tStyle ):
    #print ( tStyle )
    CheckAssert ( bBool = ( len ( tStyle ) == 6 ), sMsg = "Invalid Shape of <tStyle>!", sExtraInfo = str ( tStyle ) )
    
    tValidMarker = ( ".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*", "h", "H", "+", "x",
                     "D", "d", "|", "_", "P", "X", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, "None", "none", " ", "" )

    tValidLinestyle = ( "-", "--", "-.", ":", "none", "None", " ", "" )
    
    uColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, uLabel = tStyle
    CheckAssert ( bBool = ( sMarker in tValidMarker ), sMsg = "Invalid Parameter <sMarker>!" )
    CheckAssert ( bBool = ( sLineStyle in tValidLinestyle ), sMsg = "Invalid Parameter <sLineStyle>!" )
    CheckAssert ( bBool = ( isinstance ( uLabel, ( str, tuple, list ) ) ), sMsg = "Invalid Parameter <uLabel>! ")
    CheckAssert ( bBool = ( isinstance ( uColor, ( str, list, tuple ) ) ), sMsg = "Invalid Parameter <uColor>!" )
    
    return ( uColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, uLabel )

def StyleCheck4 ( tStyle ):
    CheckAssert ( bBool = ( len ( tStyle ) == 4 ), sMsg = "Invalid Shape of <tStyle>!", sExtraInfo = str ( tStyle ) )
    
    tValidMarker = ( ".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*", "h", "H", "+", "x",
                      "D", "d", "|", "_", "P", "X", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, "None", "none", " ", "" )
    
    sColor, sMarker, fMarkerSize, sLabel = tStyle
    CheckAssert ( bBool = ( sMarker in tValidMarker ), sMsg = "Invalid Parameter <sMarker>!" )
    CheckAssert ( bBool = ( type ( sLabel == str ) ), sMsg = "Invalid Parameter <sLabel>! ")
    CheckAssert ( bBool = ( type ( sColor == str ) ), sMsg = "Invalid Parameter <sColor>!" )
    
    return ( sColor, sMarker, fMarkerSize, sLabel )

def StyleCheck5 ( tStyle ):
    CheckAssert ( bBool = ( len ( tStyle ) == 5 ), sMsg = "Invalid Shape of <tStyle>!", sExtraInfo = str ( tStyle ) )
    
    tValidMarker = ( ".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*", "h", "H", "+", "x",
                      "D", "d", "|", "_", "P", "X", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, "None", "none", " ", "" )
    
    sColor, sMarker, fMarkerSize, sLabel, sEdgeColor = tStyle
    CheckAssert ( bBool = ( sMarker in tValidMarker ), sMsg = "Invalid Parameter <sMarker>!" )
    CheckAssert ( bBool = ( type ( sLabel == str ) ), sMsg = "Invalid Parameter <sLabel>! ")
    CheckAssert ( bBool = ( type ( sColor == str ) ), sMsg = "Invalid Parameter <sColor>!" )
    CheckAssert ( bBool = ( type ( sEdgeColor == str ) ), sMsg = "Invalid Parameter <sEdgeColor>!" )
    
    return ( sColor, sMarker, fMarkerSize, sLabel, sEdgeColor )

def PlotStyleCheck ( iDim, tStyle ):
    tStyles = None
    
    # für jeden Plot ein eigenes Style-Sheet
    if ( iDim == len ( tStyle ) ): 
        for ik in range ( iDim ): # sicherstellen, dass in keinem der Unter-Style-Sheets tStyle[ ik ] eine Farbpallete auftaucht
            CheckAssert ( bBool = ( ( len ( tStyle[ ik ] ) == 6 ) and ( tStyle[ ik ][ 0 ] not in plt.colormaps () )), sMsg = "Shape Mismatch (len[tSyle] != iDim) !" )
        tStyles = tStyle   
        
    # nur eine Plot-Serie
    elif ( ( iDim == 1 ) and ( len ( tStyle ) == 6 ) ):
        tStyles = tuple ( tStyle )
        
    # Farben und Label müssen generiert werden
    else: 
        uColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, uLabel = StyleCheck6 ( tStyle )            
        ListStyles = list ()
        
        if ( uColor in plt.colormaps () ):
            uColors = CreateDiscreteColorMap ( sColorMapName = uColor, iNumEntries = iDim )
        else:
            uColors = uColor
            
        for ik in range ( iDim ):
            if ( type ( uLabel ) == str ):
                sLabel = uLabel
            else:
                sLabel = uLabel[ ik ]
            if ( type ( uColors ) == str ):
                sColor = uColor  
            else:
                sColor = uColors[ ik ]
            
            ListStyles.append ( ( sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, sLabel ) )
            
            tStyles = tuple ( ListStyles )
            
    return ( tStyles )

# ********************** Allgemeine Funktion zur Darstellung von Annotationen basierend auf einer übergebenen Liste ************************
###                      Format eines einzelnen Listeneintrags:  ( sText, X-Koordinate, Y-Koordinate, fFontSize, optional: sColor  )
def DrawAnnotations ( ListAnnotation, GraphicConfig, CAxis = None ):
    CheckAssert ( bBool = ( len ( ListAnnotation[ -1 ] ) in [ 4, 5 ] ), sMsg = "Invalid Shape for <ListAnnotation>!",
                  sExtraInfo = "Length: %d" % ( len ( ListAnnotation ) ) )
    for ij in range ( len ( ListAnnotation ) ):
        if ( len ( ListAnnotation[ -1 ] ) == 5 ):
            sText, fPosX, fPosY, fFontSize, sColor = ListAnnotation[ ij ]
        elif ( len ( ListAnnotation[ -1 ] ) == 4 ):
            sText, fPosX, fPosY, fFontSize = ListAnnotation[ ij ]
            sColor = "black"
        if ( CAxis is not None ):
            CAxis.annotate ( sText, ( fPosX, fPosY ), ha = GraphicConfig.sAnnotationHorzAlign, va = GraphicConfig.sAnnotationVertAlign, 
                             fontname = GDictPlotParameter.get ( "FontName" ), fontsize = fFontSize, color = GetColor ( sColor ) )
        else:
            plt.annotate ( sText, ( fPosX, fPosY ), ha = GraphicConfig.sAnnotationHorzAlign, va = GraphicConfig.sAnnotationVertAlign, 
                           fontname = GDictPlotParameter.get ( "FontName" ), fontsize = fFontSize, color = GetColor ( sColor ) )
                
    return

# ********************** Allgemeine Funktion zur Darstellung von Markierungen basierend auf einer übergebenen Liste ************************
###                      Format eines einzelnen Listeneintrags:  
###                                 ( sMarker, X-Koordinate, Y-Koordinate, fMarkerSize, sMarkerFaceColor, optional sEdgecolor  )
def DrawMarker ( ListMarker, GraphicConfig, CAxis = None ):
    CheckAssert ( bBool = ( len ( ListMarker[ -1 ] ) in [ 5, 6, 7 ] ), sMsg = "Invalid Shape for <ListAnnotation>!",
                  sExtraInfo = "Length: %d" % ( len ( ListMarker ) ) )
    for ij in range ( len ( ListMarker ) ):
        if ( len ( ListMarker[ -1 ] ) == 5 ):
            sMarker, fPosX, fPosY, fMarkerSize, sFaceColor = ListMarker[ ij ]
            sEdgeColor = sFaceColor
            fMarkerEdgeWidth = 2.0
        elif ( len ( ListMarker[ -1 ] ) == 6 ):
            sMarker, fPosX, fPosY, fMarkerSize, sFaceColor, sEdgeColor = ListMarker[ ij ]
            fMarkerEdgeWidth = 2.0
        elif ( len ( ListMarker[ -1 ] ) == 7 ):
            sMarker, fPosX, fPosY, fMarkerSize, sFaceColor, sEdgeColor, fMarkerEdgeWidth = ListMarker[ ij ]
        ### ALLE Marker bekommen EIN Label
        if ( ( ij == ( len ( ListMarker ) - 1 ) ) and ( GraphicConfig.sMarkerSingleLabel ) ):
            sLabel = GraphicConfig.sMarkerSingleLabel
        else:
            sLabel = None
            
        if ( CAxis is not None ):
            CAxis.plot ( fPosX, fPosY, marker = sMarker, markersize = fMarkerSize, markerfacecolor = GetColor ( sFaceColor ), 
                         markeredgewidth = fMarkerEdgeWidth, color = GetColor ( sEdgeColor ), linestyle = "", label = sLabel )
        else:
            plt.plot ( fPosX, fPosY, marker = sMarker, markersize = fMarkerSize, markerfacecolor = GetColor ( sFaceColor ), 
                       markeredgewidth = fMarkerEdgeWidth, color = GetColor ( sEdgeColor ), linestyle = "", label = sLabel )
                
    return
 
"""
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Test-Druck ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tStyle = ( ( "r12", "H", 0.0, "-.", 3.0, "bla1" ), ( "s12", "X", 0.0, "-", 3.0, "bla2" ), 
          ( "s12", ".", 12.0, "--", 3.0, "bla3" ) ) 

#tStyle = ( "viridis", "o", 20.0, "--", 3.0, "blabla" )
HLine = pl.CLine ( sLineColor = "p12", fLinePos = 0.25, fLineWidth = 4.0, sLineStyle = "-", sLineLabel = "test" ) 
CBox = pl.CInfoBox( sText = "halle", fBoxPosX = 1.5, fBoxPosY = 1.5, iBoxFontSize = 22, sFaceColor = "azure", fAlpha = None )

aX = np.linspace ( start = 0.0, stop = 2 * np.pi, num = 100 )
aY = np.zeros ( shape = ( 3, aX.shape[ 0 ] ), dtype = np.float32 )

aY[ 0, : ] = np.cos ( aX )
aY[ 1, : ] = np.sin ( aX )
aY[ 2, : ] = np.cos ( aX ) + np.sin ( aX )

CGrCo = pl.CGraphicConfig ( sTitle = "Testdruck", sLabelX = "X-Koordinate", sLabelY  = "Y-Koordinate", HLine1 = HLine, InfoBox = CBox )
tZoom = ( [ 2, 3], [ -1.0, -0.5], 2, 1 )
pl.PlotXnY ( aX = aX, aY = aY, GraphicConfig = CGrCo, tStyles = tStyle, tZoom = tZoom, bShowLegend = True )

"""
##  tStyles: Tupel ( Color/Colormap, Marker, MarkerSize, LineStyle, LineWidth, Label/Label-Tuple ) 
## ODER für jede Plot-Serie je ein Tupel ( Color, Marker, MarkerSize, LineStyle, LineWidth, Label ); wird von PlotStyleCheck geprüft 
## tZoom = ( tXRange, tYRange, iZoomFactor, Position )
## tZoom[ 3 ] = loc : 'upper right' : 1, 'upper left' : 2, 'lower left' : 3, 'lower right' : 4, 'right' : 5, 'center left'  : 6,
## 'center right' : 7, 'lower center' : 8, 'upper center' : 9, 'center' : 10
def PlotXnY ( aX, aY, GraphicConfig, tStyles, tZoom = None, tVLine = None, tHLine = None, tVLine2 = None, tHLine2 = None, bShowLegend = False ):    
## geändert: 31.01.2026; Übergabe der Plot-Stile als Tupel mit umfangreichen Checks und Generierung der Stile bei Mehrfach-Plots durch
## eigen Funktgion PlotStyleCheck
    fig = plt.figure ( figsize = GraphicConfig.tFigureSize )
    
    if ( isinstance ( aY, ( list, tuple ) ) ):
        aY = np.asarray ( aY, dytpe = np.float64 )
        
    tStyle = PlotStyleCheck ( aY.shape[ 0 ], tStyles )
    
    if ( aY.ndim > 1 ):
        CheckAssert ( bBool = ( len ( tStyle ) == aY.shape[ 0 ] ), sMsg = "Shape Mismatch <tStyle>!" )
    else:
        CheckAssert ( bBool = ( len ( tStyle ) == 6 ), sMsg = "Invalid Parameter Shape <tStyle>!" )
    
    if ( tZoom is not None ):
        CheckAssert ( bBool = ( len ( tZoom ) == 4 ), sMsg = "Shape Mismatch <tZoom>!" )
        
    if ( tZoom is not None ):
        CAxis = plt.axes ()

    if ( aY.ndim > 1 ):
        for ik in range ( aY.shape[ 0 ] ):
            sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, sLabel = tStyle[ ik ]
            
            if ( sMarker ):
                plt.plot ( aX, aY[ ik ], marker = sMarker, markersize = fMarkerSize, linewidth = fLineWidth, linestyle = sLineStyle, 
                           label = sLabel, color = GetColor ( sColor ), alpha = 0.5 )
            else:
                plt.plot ( aX, aY[ ik ], linewidth = fLineWidth, linestyle = sLineStyle, label = sLabel, color = GetColor ( sColor ) )        
    else:
        sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, sLabel = tStyle
        if ( sMarker ):
            plt.plot ( aX, aY, marker = sMarker, markersize = fMarkerSize, linewidth = fLineWidth, linestyle = sLineStyle, 
                       label = sLabel, color = GetColor ( sColor ) )
        else:
            plt.plot ( aX, aY, linewidth = fLineWidth, linestyle = sLineStyle, label = sLabel, color = GetColor ( sColor ) )        
    
    GraphicConfig.Set ( CFigure = fig )
    DrawFrame ( GraphicConfig = GraphicConfig, bShowLegend = bShowLegend )

    if ( tZoom is not None ):
        CAxisIn = zoomed_inset_axes ( parent_axes = CAxis, zoom = tZoom[ 2 ], loc = tZoom[ 3 ] ) 
        if ( aY.ndim > 1 ):
            sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, sLabel = tStyle[ 0 ]
            CAxisIn.plot ( aX, aY[ 0, :], marker = sMarker, markersize = fMarkerSize, linewidth = fLineWidth, linestyle = sLineStyle, 
                           color = GetColor ( sColor ) )
        else:
            sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, sLabel = tStyle
            CAxisIn.plot ( aX, aY, marker = sMarker, markersize = fMarkerSize, linewidth = fLineWidth, linestyle = sLineStyle, 
                           color = GetColor ( sColor ) )
            
        CAxisIn.set_xlim ( tZoom[ 0 ] )
        CAxisIn.set_ylim ( tZoom[ 1 ] )
        plt.xticks ( visible = False )
        plt.yticks ( visible = False )
        if ( GraphicConfig.sGridAxis in [ "x", "y", "both" ] ):
            plt.grid ( visible = True, axis = GraphicConfig.sGridAxis )
        mark_inset ( parent_axes = CAxis, inset_axes = CAxisIn, loc1 = 1, loc2 = 4, fc = "none", ec = "0.6" )
     
    plt.show ()
    
    return

# ********************* Polar-Plot (theta; Rho) mit einem Kreuz zur Kennzechnung der maximalen und minimalen Ausdehnung ********************
#### gute Wahl : tStyleScatter = ( "o", 40, "b12" ), tCrossStyle  = ( 3.0, "o12", 2.0, "o10" )
def PlotPolar ( aRad, aRho, tStyleScatter, tCrossStyle, GraphicConfig ):
    CFig = plt.figure ( figsize = GraphicConfig.tFigureSize )
    CAxis = CFig.add_subplot ( projection = "polar" )

    aTheta = np.append ( aRad, aRad[ 0 ] )
    aR = np.append ( aRho, aRho[ 0 ])

    CAxis.plot ( aTheta, aR, linewidth = 2.0, color = GetColor ( "b5" ) )
    CAxis.fill ( aTheta, aR, alpha = 0.2, facecolor = GetColor ( "b5" ) )

    sMarker, fMarkerSize, sMarkerColor = tStyleScatter
    CAxis.scatter ( aRad, aRho, marker = sMarker, s = fMarkerSize, color = GetColor ( sMarkerColor ), alpha = 1.0 )
    CAxis.set_ylim ( [ 0.0, np.amax ( aRho ) * 1.05 ] )
    CAxis.set_rlabel_position ( GraphicConfig.fPosVariable )

    fRho_max, iRho_index_max = np.amax ( aRho ), np.argmax ( aRho )
    fRad_max = aRad[ iRho_index_max ]
    fRad_min = fRad_max + np.pi / 2.0

    if ( tCrossStyle is not None ):
        fCrossLineWidthMax, sCrossColorMax, fCrossLineWidthMin, sCrossColorMin = tCrossStyle 
        CAxis.plot ( [ fRad_max, fRad_max ], [ 0,  fRho_max ], linestyle = "--", linewidth = fCrossLineWidthMax, 
                       color = GetColor ( sCrossColorMax ) )
        CAxis.plot ( [ fRad_max + np.pi, fRad_max + np.pi ], [ 0,  fRho_max ], linestyle = "--", linewidth = fCrossLineWidthMax, 
                       color = GetColor ( sCrossColorMax ) )

        CAxis.plot ( [ fRad_min, fRad_min ], [ 0,  fRho_max ], linestyle = ":", linewidth = fCrossLineWidthMin, 
                       color = GetColor ( sCrossColorMin ) )
        CAxis.plot ( [ fRad_min + np.pi, fRad_min + np.pi ], [ 0,  fRho_max ], linestyle = ":", linewidth = fCrossLineWidthMin, 
                       color = GetColor ( sCrossColorMin ) )

    CAxis.set_title ( label = GraphicConfig.sTextTitle, fontname = GDictPlotParameter.get ( "FontName" ), 
                      fontsize = GDictPlotParameter.get ( "TitleSize" ) )

    for CLabel in CAxis.get_ymajorticklabels ():
        CLabel.set_color ( GetColor ( "g18" ) )
        CLabel.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
        CLabel.set_fontsize ( fontsize = GDictPlotParameter.get ( "TickSize" ) )
    
    for CLabel in CAxis.get_xmajorticklabels ():
        CLabel.set_color ( "black" )
        CLabel.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
        CLabel.set_fontsize ( fontsize = GDictPlotParameter.get ( "TickSize" ) )

    CAxis.grid ( visible = True )
    plt.show ()
    
    return

# **************************** Plot (X; Y) mit einer farbkodierten Z-Variable und ggf. einer Umrandung *************************************
# ************* tStyle =  sColorMap, sMarker, fMarkerSize, sLabel 
# ************* tBorder = ( aBorder, Color, LineStyle, LineWidth )
# ************* Colorbar wird über das Attribut sTextLegend der Klasse GraphicConfig aktiviert
## tStyle =  sColor, sMarker, fMarkerSize, sLabel 
## der Colorbar wird über das Attribut sTextLegend der Klasse GraphicConfig aktiviert
def PlotScatterXY ( aX, aY, aZ, GraphicConfig, tStyle, sEdgeColor = "none", ListMarker = None, tBorder = None, tVLine = None, tHLine = None, 
                    tVLine2 = None, tHLine2 = None, bShowLegend = False ):
    CheckAssert ( bBool = ( aX.ndim == 1 and aY.ndim == 1 ), sMsg = "aX and aY must be 1-dimensional!" )
    CheckAssert ( bBool = ( ( aX.shape == aY.shape ) and ( aX.shape == aZ.shape ) ), sMsg = "Dimension mismatch!" )
    CheckAssert ( bBool = ( len ( tStyle ) == 4 ), sMsg = "<tStyle> must be 4-dimensional!" )
    
    if ( tBorder is not None ):
        CheckAssert ( bBool = ( len ( tBorder ) == 4 ), sMsg = "<tStyle> must be 3-dimensional!" )
        aBorder, sBorderColor, sBorderStyle, fBorderLineWidth = tBorder
        CheckAssert ( bBool = ( aBorder.ndim == 2 ), sMsg = "aBorder must be 2-dimensional" )
        
    sColor, sMarker, fMarkerSize, sLabel  = tStyle
    
    CFigure, CAxis = plt.subplots ( nrows = 1, ncols = 1, figsize = GraphicConfig.tFigureSize )
    if ( sColor in plt.colormaps () ):
        CPathCollection = CAxis.scatter ( x = aX, y = aY, s = fMarkerSize, marker = sMarker, c = aZ, edgecolor = GetColor ( sEdgeColor), 
                                          cmap = sColor, label = sLabel )
    else:
        CPathCollection = CAxis.scatter ( x = aX, y = aY, s = fMarkerSize, marker = sMarker, c = GetColor ( sColor ), 
                                          edgecolor = GetColor ( sEdgeColor), cmap = None, label = sLabel )
        
    if ( ListMarker is not None ):
        DrawMarker ( ListMarker = ListMarker, GraphicConfig = GraphicConfig, CAxis = None )
        
    if ( GraphicConfig.sTextLegend ):
        CAxisDivider = make_axes_locatable ( CAxis )
        CAxis2 = CAxisDivider.append_axes ( "right", size = "3%", pad = GraphicConfig.fPosVariable )
        CColorBar = CFigure.colorbar ( mappable = CPathCollection, cax = CAxis2 )
    
        CColorBar.ax.set_ylabel ( GraphicConfig.sTextLegend, fontname = GDictPlotParameter.get ( "FontName" ), 
                                  fontsize = GDictPlotParameter.get ( "LabelSizeColorbar" ), rotation = -90, verticalalignment = "bottom" )

        for CLabel in CColorBar.ax.get_yticklabels ():
            CLabel.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
            CLabel.set_fontsize ( fontsize = GDictPlotParameter.get ( "TickSizeColorbar" ) )
    
    if ( tBorder is not None ):
        CAxis.plot ( aBorder[:, 0 ], aBorder[ :, 1 ], color = GetColor ( sBorderColor ), linestyle = sBorderStyle, linewidth = fBorderLineWidth )
    
    if ( bShowLegend == True ):
        CAxis.legend ( loc = GraphicConfig.sLegendPosition,
                       prop = { "family": GDictPlotParameter.get ( "FontName" ), "size": GDictPlotParameter.get ( "LegendSize" ) } )
         
    DrawFrameAxis ( CAxis = CAxis, GraphicConfig = GraphicConfig, iIndex = 0, sGridAxis = GraphicConfig.sGridAxis )
    plt.show ()

    return

# ************************** Plot einer Karte als Hintergrund zu einem Scatter-Plot ************************************
#   	               tStyleGeoDataEdge = ( sGeoDataEdgeColor, fGeoDataEdgeLineWidth )
#                          tMarker = ( fX, fY, sMarker, fMarkerSize, sColor )
def PlotScatterwMap ( aX, aY, aZ, tMarker, GraphicConfig, tStyle, DfGeoData, tStyleGeoDataEdge, sGeoDataFaceColor = "none", 
                      tStyleRectangle = None, fAlpha = 1.0 ):
    CheckAssert ( bBool = ( aX.ndim == 1 and aY.ndim == 1 ), sMsg = "aX and aY must be 1-dimensional!" )
    CheckAssert ( bBool = ( ( aX.shape == aY.shape ) and ( aX.shape == aZ.shape ) ), sMsg = "Dimension mismatch!" )
    CheckAssert ( bBool = ( len ( tStyle ) == 4 ), sMsg = "<tStyle> must be 4-dimensional!" )    
    CheckAssert ( bBool = ( len ( tStyleGeoDataEdge ) == 2 ), sMsg = "<tStyleGeoDataEdge> must be 2-dimensional!" )    
    CheckAssert ( bBool = ( isinstance ( DfGeoData, GeoDataFrame ) ), sMsg = "Wrong Format <DfGeoData>!" )
        
    sColor, sMarker, fMarkerSize, sLabel  = tStyle
    sGeoDataEdgeColor, fGeoDataEdgeLineWidth = tStyleGeoDataEdge
    
    CFigure, CAxis = plt.subplots ( nrows = 1, ncols = 1, figsize = GraphicConfig.tFigureSize )

    DfGeoData.plot ( kind = "geo", ax = CAxis, edgecolor = GetColor ( sGeoDataEdgeColor ), linewidth = fGeoDataEdgeLineWidth, 
                     facecolor = GetColor ( sGeoDataFaceColor ), alpha = fAlpha )
    
    if ( tStyleRectangle is not None ):
        CheckAssert ( bBool = ( len ( tStyleRectangle ) == 8 ), sMsg = "Invalid Shape of <tStyleRectangle>!",
                     sExtraInfo = "fLowerLeftX, fLowerLeftY, fWidth, fHeight, fLineWidth, sLineStyle, sEdgeColor, sFaceColor" )
        fLowerLeftX, fLowerLeftY, fWidth, fHeight, fLineWidth, sLineStyle, sEdgeColor, sFaceColor = tStyleRectangle
        CRect = Rectangle ( ( fLowerLeftX, fLowerLeftY ), fWidth, fHeight, linewidth = fLineWidth, linestyle = sLineStyle, 
                              edgecolor = GetColor ( sEdgeColor) , facecolor = GetColor ( sFaceColor ) )

        CAxis.add_patch ( CRect )
    
    if ( sColor in plt.colormaps () ):
        CPathCollection = CAxis.scatter ( x = aX, y = aY, s = fMarkerSize, marker = sMarker, c = aZ, cmap = sColor, label = sLabel )
    else:
        CPathCollection = CAxis.scatter ( x = aX, y = aY, s = fMarkerSize, marker = sMarker, c = GetColor ( sColor ), cmap = None, label = sLabel )
        
    if ( tMarker is not None ):
        for ik in range ( len ( tMarker ) ):
            fX, fY, sMarker, fMarkerSize, sColor = tMarker[ ik ] 
            CAxis.scatter ( x = fX, y = fY, s = fMarkerSize, marker = sMarker, color = GetColor ( sColor ) )#
                            #edgecolor = GetColor ( sColor ), facecolor = "none", cmap = None )
             
    if ( GraphicConfig.sTextLegend ):
        CAxisDivider = make_axes_locatable ( CAxis )
        CAxis2 = CAxisDivider.append_axes ( "right", size = "3%", pad = GraphicConfig.fPosVariable )
        CColorBar = CFigure.colorbar ( mappable = CPathCollection, cax = CAxis2 )
        
        CColorBar.ax.set_ylabel ( GraphicConfig.sTextLegend, fontname = GDictPlotParameter.get ( "FontName" ), 
                                  fontsize = GDictPlotParameter.get ( "LabelSizeColorbar" ), rotation = -90, verticalalignment = "bottom" )

        for CLabel in CColorBar.ax.get_yticklabels ():
            CLabel.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
            CLabel.set_fontsize ( fontsize = GDictPlotParameter.get ( "TickSizeColorbar" ) )
        
    DrawFrameAxis ( CAxis = CAxis, GraphicConfig = GraphicConfig, iIndex = 0, sGridAxis = GraphicConfig.sGridAxis )
    

    plt.show ()
    
    return
# ***************************** Scatter Plot with colored Decison Lines used by Classification Algorithms **********************************
def PlotScatterwContour ( aX, aY, FuncPredict, GraphicConfig, sColorMap, iNumGridPoints = 300 ):
## geändert 29.01.206: Umstellung auf GraphicConfig anstzelle von PlotOption 
    plt.figure ( figsize = GraphicConfig.tFigureSize )

    fEpsilon = 0.1
    
    aX0, aX1 = aX[ :, 0 ], aX[ :, 1 ]

    fX_min, fX_max = np.min ( aX0 ) - fEpsilon, np.max ( aX0 ) + fEpsilon
    fY_min, fY_max = np.min ( aX1 ) - fEpsilon, np.max ( aX1 ) + fEpsilon
    aXX, aYY = np.meshgrid ( np.linspace ( start = fX_min, stop = fX_max, num = iNumGridPoints, endpoint = True ), 
                             np.linspace ( start = fY_min, stop = fY_max, num = iNumGridPoints, endpoint = True ) )

    aZ = FuncPredict ( np.c_[ np.ravel ( aXX ), np.ravel ( aYY ) ] )
        
    aZ = np.reshape ( aZ, shape = aXX.shape )
        
    DrawFrame ( GraphicConfig = GraphicConfig )

    plt.contourf ( aXX, aYY, aZ, cmap = sColorMap, alpha = 0.8 )
    plt.scatter ( aX0, aX1, c = aY, cmap = sColorMap, s = 100, edgecolors = "k" )

    plt.show ()
    
    return
## *************************************** Darstellung drei-dimensionaler Daten als Contour-Plot *******************************************
# Colorbar nur dann, wenn die Beschriftung via sTextLegend aktiviert wird
# Der sColor-Eintrag von tStyleDataObs bestimmt, ob die Beobachtungen in der Farbpalette des Kontour-Plots ( sColor = "") 
# oder als einfarbige Markierungen dargestellt werden ( sColor != "" ) 
def PlotContour ( aX, aY, aData2D, iNumLevel, GraphicConfig, sColorMap, aDataObserved = None, tStyleDataObs = None, 
                  tLimX = None, tLimY = None ):
    CheckAssert ( bBool = ( aData2D.shape == ( aY.shape[ 0 ], aX.shape[ 0 ] ) ), sMsg = "Shape Mismatch!",
                 sExtraInfo = "Shape Data2D: %s, Shape aX: %s, Shape aY: %s" % ( str ( aData2D.shape ), str ( aX.shape ), str ( aY.shape ) ) )
    if ( aDataObserved is not None ):
        CheckAssert ( bBool = ( ( isinstance ( aDataObserved, np.ndarray ) ) and ( aDataObserved.shape[ 0 ] == 3 ) ), 
                      sMsg = "Wrong Format <uDataObserved>!" )
        
    CheckAssert ( bBool = ( tStyleDataObs is not None ), sMsg = "<tStyleDataObs> can't be None!" )

    uColor, sMarker, fMarkerSize, sLabel, sEdgeColor = StyleCheck5 ( tStyleDataObs )
    
    iNumColorBarTicks = 10
    fMax, fMin = np.amax ( aData2D ), np.amin ( aData2D )
    if ( fMax == fMin ):
        print ( ">> PlotContour > Constant Field!" )
        print ( fMax, fMin )
        return
    
    aContourLevel = np.linspace ( start = fMin, stop = fMax, num = iNumLevel )
    
    ListLabel = list ()
    iStepSize = int ( np.ceil ( float ( iNumLevel ) / float ( iNumColorBarTicks ) ) )

    for ik in range ( 0, iNumLevel, iStepSize ):
        ListLabel.append ( np.around ( aContourLevel[ ik ], 1 ) )
    
    CFigure, CAxis = plt.subplots ( figsize = GDictPlotParameter.get ( "FigureSize" ), sharey = True )
    # aus einem schwer nachvollziehbaren Grund steht das 2D-Array auf dem Kopf, wenn es sich um ein Bild handelt.
    ## Daher anfänglich die Nutzung von flipud, also np.flipud ( aData2d ). Da das bei anderen 2D-Arrays aber nicht gilt, 
    ## wieder rausgenommen am 16.04.2026
    
    #print ( ">>>>>> X", np.amin ( aX ), np.amax ( aX ), aX.shape, aData2D.shape )
    #print ( ">>>>>> Y", np.amin ( aY ), np.amax ( aY ), aY.shape, aData2D.shape )
    
    ## 19.05.2026 nach vielem Probieren passt diese Formatierung; erst y dann x
    CContourPlot = CAxis.contourf ( aY, aX, np.transpose ( aData2D ), cmap = sColorMap, levels = aContourLevel, alpha = 0.8, origin = "upper" )
    #CAxis.clabel ( CContourPlot, fontsize = 20 ) # Annotations! Sieht aber nicht gut aus 
    
    if ( aDataObserved is not None ):
        if ( uColor in plt.colormaps () ):
            CAxis.scatter ( x = aDataObserved[ 1, : ], y = aDataObserved[ 0, : ], marker = sMarker, s = fMarkerSize, c = aDataObserved[ 2, : ], 
                            edgecolors = GetColor ( sEdgeColor ), cmap = uColor )
        else:
            CAxis.plot ( aDataObserved[ 1, : ], aDataObserved[ 0, : ], color = GetColor ( uColor ), marker = sMarker, 
                         markersize = fMarkerSize, linestyle = "", linewidth = 0.0, alpha = 0.7 )
            ## das Bild passt nicht zum Rahmen des Plots, wenn die Shapes nicht passen
            
        if ( ( tLimX is not None ) and ( len ( tLimX ) == 2 ) ):
            CAxis.set_ylim ( np.amin ( aX ) - tLimY[ 0 ], np.amax ( aX ) + tLimY[ 1 ] )
        if ( ( tLimY is not None ) and ( len ( tLimY ) == 2 ) ):
            CAxis.set_xlim ( np.amin ( aY ) - tLimX[ 0 ], np.amax ( aY ) + tLimX[ 1 ] )

    if ( GraphicConfig.sTextLegend ):
        CColorBar = CAxis.figure.colorbar ( CContourPlot, ax = CAxis, ticks = ListLabel )
        CColorBar.ax.set_ylabel ( GraphicConfig.sTextLegend, fontname = GDictPlotParameter.get ( "FontName" ), 
                                  fontsize = GDictPlotParameter.get ( "LabelSizeColorbar" ), rotation = -90, verticalalignment = "bottom" )
    
        for CLabel in CColorBar.ax.get_yticklabels ():
            CLabel.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
            CLabel.set_fontsize ( fontsize = GDictPlotParameter.get ( "TickSizeColorbar" ) )
     
    #CFigure.set_layout_engine ( "tight" )
    DrawFrameAxis ( CAxis = CAxis, GraphicConfig = GraphicConfig, iIndex = 0, sGridAxis = "both" )
    
    if ( sLabel ):
        plt.legend ( [ sLabel ], prop = { "family": GDictPlotParameter.get ( "FontName" ), "size": GDictPlotParameter.get ( "LegendSize" ) }, 
                     loc = GraphicConfig.sLegendPosition )
    
    plt.show ()

    return

def PlotBarChart ( aX, aData, GraphicConfig, uColor, sEdgeColor = None, ListAnnotation = None, fWidth = None, 
                   tRangeX = None, tRangeY = None, iMaxNumTicksY = None ):
    plt.figure ( figsize = GraphicConfig.tFigureSize )
    
    if ( aX is None ):
        aX = np.arange ( start = 1, stop = aData.shape[ 0 ] + 1 )
        aTickLabels = aX
    else:
        aTickLabels = None
        
    if ( fWidth is None ):
        if ( isinstance ( aX[ -1 ], np.datetime64 ) ):
            fWidth = aX[ -1 ] - aX[ -2 ]
        else:
            fWidth = 0.8
    
    if ( sEdgeColor is None ):
        sEdgeColor = uColor
        
    DrawFrame ( GraphicConfig = GraphicConfig, bShowLegend = False )
    
    if ( ( isinstance ( uColor, ( tuple, list, np.ndarray ) ) == True ) and ( aData.shape[ 0 ] == uColor.shape[ 0 ] ) ):
        plt.bar ( x = aX, height = aData, tick_label = aTickLabels, width = fWidth, color = uColor, edgecolor = sEdgeColor )
    elif ( type ( uColor ) == str ):
        plt.bar ( x = aX, height = aData, tick_label = aTickLabels, width = fWidth, color = GetColor ( uColor ), edgecolor = GetColor ( sEdgeColor ) )
    
    if ( ListAnnotation is not None ):
        DrawAnnotations ( ListAnnotation = ListAnnotation, GraphicConfig = GraphicConfig )
    
    if ( tRangeX is not None ):
        plt.xlim ( tRangeX[ 0 ] - 0.5, tRangeX[ 1 ] + 0.5 )
    if ( tRangeY is not None ):
        plt.ylim ( tRangeY[ 0 ], tRangeY[ 1 ] )
        
    if ( iMaxNumTicksY is not None ):
        plt.locator_params ( axis = "y", nbins = iMaxNumTicksY )
    
    if ( GraphicConfig.bHideTicksX == True ):
        plt.xticks ( [] ) 
    if ( GraphicConfig.bHideTicksY == True ):
        plt.yticks ( [] ) 
         
    if ( ( GraphicConfig.sTextLegend ) and ( GraphicConfig.sTextLegend2 ) ):
        plt.legend ( [ GraphicConfig.sTextLegend, GraphicConfig.sTextLegend2 ], 
                       prop = { "family": GDictPlotParameter.get ( "FontName" ), "size": 12 }, loc = GraphicConfig.sLegendPosition )
    
    if ( GraphicConfig.bShowPlot == True ):
        plt.show ()
    
    return

# ***************************************** Plot Funktionen mit Fläche unterhalb des Graphen ***********************************************
def PlotX2YSubplotwPatch ( Func1, Func2, tFuncIntervallX, tSupportPoints, aTicksX, aColorIndices, GraphicConfig ):
## geändert 29.01.206: Umstellung auf GraphicConfig anstzelle von PlotOption 
    aX = np.linspace ( start = tFuncIntervallX[ 0 ], stop = tFuncIntervallX[ 1 ], num = 60, endpoint = True )
    aY1 = Func1 ( aX )
    aY2 = Func2 ( aX )    
    
    plt.figure ( figsize = ( 9, 10 ) )
   
    ax1 = plt.subplot ( 2, 1, 1 )
    ax2 = plt.subplot ( 2, 1, 2 )
    ax1.grid ( visible = True, axis = "both" )
    ax2.grid ( visible = True, axis = "both" )
    ax1.set_title ( GraphicConfig.sTextTitle, fontname = GDictPlotParameter.get ( "FontName" ), fontsize = 18 )
    ax1.set_ylabel ( GraphicConfig.sTextLabelY, fontname = GDictPlotParameter.get ( "FontName" ), fontsize = 16 )
    if ( GraphicConfig.sTextTitle2 ):
        ax2.set_title ( GraphicConfig.sTextTitle2, fontname = GDictPlotParameter.get ( "FontName" ), fontsize = 18 )
    ax2.set_xlabel ( GraphicConfig.sTextLabelX, fontname = GDictPlotParameter.get ( "FontName" ), fontsize = 16 )
    ax2.set_ylabel ( GraphicConfig.sTextLabelY2, fontname = GDictPlotParameter.get ( "FontName" ), fontsize = 16 )
    plt.xticks ( fontname = GDictPlotParameter.get ( "FontName" ), fontsize = 12 )
    plt.yticks ( fontname = GDictPlotParameter.get ( "FontName" ), fontsize = 12 )
        
    for label in ax1.get_xticklabels ():
        label.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
        label.set_fontsize ( fontsize = 12 )
        
    for label in ax1.get_yticklabels ():
        label.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
        label.set_fontsize ( fontsize = 12 )
        
    if ( GraphicConfig.VLine1.fLinePos is not None ):
        ax1.axvline ( x = GraphicConfig.VLine1.fLinePos, color = GetColor ( GraphicConfig.VLine1.sLineColor ), 
                     ls = GraphicConfig.VLine1.sLineStyle, lw = GraphicConfig.VLine1.iLineWidth )
    if ( GraphicConfig.VLine2.fLinePos is not None ):
        ax2.axvline ( x = GraphicConfig.VLine2.fLinePos, color = GetColor ( GraphicConfig.VLine2.sLineColor ), 
                     ls = GraphicConfig.VLine2.sLineStyle, lw = GraphicConfig.VLine2.iLineWidth )

    if ( GraphicConfig.HLine1.fLinePos is not None ):
        ax1.axhline ( y = GraphicConfig.HLine1.fLinePos, color = GetColor ( GraphicConfig.HLine1.sLineColor ), 
                     ls = GraphicConfig.HLine1.sLineStyle, lw = GraphicConfig.HLine1.iLineWidth )
    if ( GraphicConfig.HLine2.fLinePos is not None ):
        ax2.axhline ( y = GraphicConfig.HLine2.fLinePos, color = GetColor ( GraphicConfig.HLine2.sLineColor ), 
                     ls = GraphicConfig.HLine2.sLineStyle, lw = GraphicConfig.HLine2.iLineWidth )

    ax1.plot ( aX, aY1, color = GetColor ( 2 ), linewidth = 3 )
    ax1.set_ylim ( bottom = 0 )
    ax2.plot ( aX, aY2, color = GetColor ( 2 ), linewidth = 3 )
    ax2.set_ylim ( bottom = 0 )

    for k in range ( len ( tSupportPoints ) - 1 ):
        sColorIndex = aColorIndices[ k ]
        aPatch_X = np.linspace ( start = tSupportPoints[ k ], stop = tSupportPoints[ k + 1 ], num = 60 )
        
        aPatch_Y1 = Func1 ( aPatch_X )
        aVerts = [ ( tSupportPoints[ k ], 0 ), *zip ( aPatch_X, aPatch_Y1 ), ( tSupportPoints[ k + 1 ], 0 ) ]
        CPoly = Polygon ( aVerts, facecolor = GetColor ( sColorIndex ), edgecolor = "1.0" )
        ax1.add_patch ( CPoly )
        
        aPatch_Y2 = Func2 ( aPatch_X )
        aVerts = [ ( tSupportPoints[ k ], 0 ), *zip ( aPatch_X, aPatch_Y2 ), ( tSupportPoints[ k + 1 ], 0 ) ]
        CPoly = Polygon ( aVerts, facecolor = GetColor ( sColorIndex ), edgecolor = "1.0" )
        ax2.add_patch ( CPoly )
    
    plt.tight_layout ()
    
    ax1.set_xticks ( tSupportPoints )
    ax1.set_xticklabels ( aTicksX ) 
    ax2.set_xticks ( tSupportPoints )
    ax2.set_xticklabels ( aTicksX ) 

    plt.show ()

    return  
# ************************** Plot (X; Y), (X; Y1, Y2), (X; Y1, Y2, Y3), (X1, Y1; X2, Y2), (X1, Y1, Y2; X2, Y3 ) ****************************
#                       tStyle = ( sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, sLabel )
def PlotXY ( aX, aY, tStyle, GraphicConfig, tFillArea = None, ListMarker = None, ListAnnotation = None, tRangeX = None, tRangeY = None ):
## geändert 14.04.2026: Legende wird jetzt angezeigt, wenn sLabel als Style ODER GrapicConfig.sTextLegend gesetzt ist
    plt.figure ( figsize = GraphicConfig.tFigureSize )
    sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, sLabel = StyleCheck6 ( tStyle )
    if ( len ( sLabel ) == 0 ):
        sLabel = GraphicConfig.sTextLegend
        
    DrawFrame ( GraphicConfig = GraphicConfig, bShowLegend = False )
    
    if ( tFillArea is not None ): # tFillArea : ( obere Grenez pro X-Wert, untere Grtenze pro X-Wert, Farbcode, Alpha-Wert)
        CheckAssert ( bBool = ( isinstance ( tFillArea, ( tuple, list ) ) and ( len ( tFillArea ) == 5 ) ), 
                      sMsg = "Invalid Shape <tFillArea>!" )
        plt.fill_between ( aX, tFillArea[ 0 ], tFillArea[ 1 ], facecolor = GetColor ( tFillArea[ 2 ] ), alpha = tFillArea[ 3 ],
                           label = tFillArea[ 4 ])
    
    if ( GraphicConfig.sStepPlotWhere != "none" ):
        plt.step ( aX, aY, color = GetColor ( sColor ), marker = sMarker, markersize = fMarkerSize, linestyle = sLineStyle,
                   where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth, label = sLabel )
    else:
        plt.plot ( aX, aY, color = GetColor ( sColor ), marker = sMarker, markersize = fMarkerSize, linestyle = sLineStyle,
                   linewidth = fLineWidth, label = sLabel )
    
    if ( ListAnnotation is not None ):
        DrawAnnotations ( ListAnnotation = ListAnnotation, GraphicConfig = GraphicConfig )
        
    if ( ListMarker is not None ):
        DrawMarker ( ListMarker = ListMarker, GraphicConfig = GraphicConfig, CAxis = None )
 
    if ( ( ( len ( str ( aX[ 0 ] ) ) > 8 ) ) and ( len ( str ( aX[ -1 ] ) ) > 8 ) ):
        plt.setp ( plt.gca ().xaxis.get_majorticklabels (), "rotation", 50, "ha", "right" )
    
    if ( tRangeX ):
        plt.xlim ( tRangeX[ 0 ], tRangeX[ 1 ] )
    if ( tRangeY ):
        plt.ylim ( tRangeY[ 0 ], tRangeY[ 1 ] )
        
    if ( len ( sLabel ) > 0 ):
        if ( GraphicConfig.tFigureSize[ 0 ] >= 10 ):
            plt.legend ( prop = { "family": GDictPlotParameter.get ( "FontName" ), "size": 12 }, loc = GraphicConfig.sLegendPosition )
        else:
            plt.legend ( prop = { "family": GDictPlotParameter.get ( "FontName" ), "size": 10 }, loc = GraphicConfig.sLegendPosition )
       
    if ( GraphicConfig.bShowPlot == True ):
        plt.show ()
    
    return 

def PlotX2Y ( aX, aY1, aY2, tStyleY1, tStyleY2, GraphicConfig, tFillArea = None, ListMarker = None, ListAnnotation = None, tRangeX = None, tRangeY = None ):
    plt.figure ( figsize = GraphicConfig.tFigureSize )
    sColor1, sMarker1, fMarkerSize1, sLineStyle1, fLineWidth1, sLabel1 = StyleCheck6 ( tStyleY1 )
    if ( len ( sLabel1 ) == 0 ):
        sLabel1 = GraphicConfig.sTextLegend
    sColor2, sMarker2, fMarkerSize2, sLineStyle2, fLineWidth2, sLabel2 = StyleCheck6 ( tStyleY2 )
    if ( len ( sLabel2 ) == 0 ):
        sLabel2 = GraphicConfig.sTextLegend2
        
    DrawFrame ( GraphicConfig = GraphicConfig, bShowLegend = False )
    
    if ( tFillArea is not None ): # tFillArea : ( obere Grenez pro X-Wert, untere Grtenze pro X-Wert, Farbcode, Alpha-Wert, Label)
        CheckAssert ( bBool = ( isinstance ( tFillArea, ( tuple, list ) ) and ( len ( tFillArea ) == 5 ) ), 
                      sMsg = "Invalid Shape <tFillArea>!" )
        plt.fill_between ( aX, tFillArea[ 0 ], tFillArea[ 1 ], facecolor = GetColor ( tFillArea[ 2 ] ), alpha = tFillArea[ 3 ], 
                           label = tFillArea[ 4 ] )
        
    if ( GraphicConfig.sStepPlotWhere != "none" ):
        plt.step ( aX, aY1, color = GetColor ( sColor1 ), marker = sMarker1, markersize = fMarkerSize1, linestyle = sLineStyle1,
                   where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth1, label = sLabel1 )
        plt.step ( aX, aY2, color = GetColor ( sColor2 ), marker = sMarker2, markersize = fMarkerSize2, linestyle = sLineStyle2,
                   where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth2, label = sLabel2 )
    else:    
        plt.plot ( aX, aY1, color = GetColor ( sColor1 ), marker = sMarker1, markersize = fMarkerSize1, linestyle = sLineStyle1,
                   linewidth = fLineWidth1, label = sLabel1 )
        plt.plot ( aX, aY2, color = GetColor ( sColor2 ), marker = sMarker2, markersize = fMarkerSize2, linestyle = sLineStyle2,
                   linewidth = fLineWidth2, label = sLabel2 )
    
    if ( ListAnnotation is not None ):
        DrawAnnotations ( ListAnnotation = ListAnnotation, GraphicConfig = GraphicConfig )
        
    if ( ListMarker is not None ):
        DrawMarker ( ListMarker = ListMarker, GraphicConfig = GraphicConfig, CAxis = None )
        
    if ( ( len ( sLabel1 ) > 0 ) or ( len ( sLabel2 ) > 0 ) ):
        plt.legend ( #[ sTextLegend, sTextLegend2 ], 
                     prop = { "family": GDictPlotParameter.get ( "FontName" ), "size": 12 }, loc = GraphicConfig.sLegendPosition )
    
    if ( ( ( len ( str ( aX[ 0 ] ) ) > 8 ) ) and ( len ( str ( aX[ -1 ] ) ) > 8 ) ):
        plt.setp ( plt.gca ().xaxis.get_majorticklabels (), "rotation", 50, "ha", "right" )
    
    if ( tRangeX ):
        plt.xlim ( tRangeX[ 0 ], tRangeX[ 1 ] )
    if ( tRangeY ):
        plt.ylim ( tRangeY[ 0 ], tRangeY[ 1 ] )
        
    if ( GraphicConfig.bShowPlot == True ):
        plt.show ()
         
    return 

def PlotX3Y ( aX, aY1, aY2, aY3, tStyleY1, tStyleY2, tStyleY3, GraphicConfig, tFillArea = None, ListMarker = None, ListAnnotation = None, 
              tRangeX = None, tRangeY = None ):   
    plt.figure ( figsize = GraphicConfig.tFigureSize )
    sColor1, sMarker1, fMarkerSize1, sLineStyle1, fLineWidth1, sLabel1 = StyleCheck6 ( tStyleY1 )
    if ( len ( sLabel1 ) == 0 ):
        sLabel1 = GraphicConfig.sTextLegend
    sColor2, sMarker2, fMarkerSize2, sLineStyle2, fLineWidth2, sLabel2 = StyleCheck6 ( tStyleY2 )
    if ( len ( sLabel2 ) == 0 ):
        sLabel2 = GraphicConfig.sTextLegend2
    sColor3, sMarker3, fMarkerSize3, sLineStyle3, fLineWidth3, sLabel3 = StyleCheck6 ( tStyleY3 )
    if ( len ( sLabel3 ) == 0 ):
        sLabel3 = GraphicConfig.sTextLegend3
        
    DrawFrame ( GraphicConfig = GraphicConfig, bShowLegend = False )
    
    if ( tFillArea is not None ): # tFillArea : ( obere Grenez pro X-Wert, untere Grtenze pro X-Wert, Farbcode, Alpha-Wert)
        CheckAssert ( bBool = ( isinstance ( tFillArea, ( tuple, list ) ) and ( len ( tFillArea ) == 5 ) ), 
                      sMsg = "Invalid Shape <tFillArea>!" )
        plt.fill_between ( aX, tFillArea[ 0 ], tFillArea[ 1 ], facecolor = GetColor ( tFillArea[ 2 ] ), alpha = tFillArea[ 3 ],
                           label = tFillArea[ 4 ] )
    
    if ( GraphicConfig.sStepPlotWhere != "none" ):
        plt.step ( aX, aY1, color = GetColor ( sColor1 ), marker = sMarker1, markersize = fMarkerSize1, linestyle = sLineStyle1,
                   where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth1, label = sLabel1 )
        plt.step ( aX, aY2, color = GetColor ( sColor2 ), marker = sMarker2, markersize = fMarkerSize2, linestyle = sLineStyle2,
                   where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth2, label = sLabel2 )
        plt.step ( aX, aY3, color = GetColor ( sColor3 ), marker = sMarker3, markersize = fMarkerSize3, linestyle = sLineStyle3,
                   where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth3, label = sLabel3 )
    else:
        plt.plot ( aX, aY1, color = GetColor ( sColor1 ), marker = sMarker1, markersize = fMarkerSize1, linestyle = sLineStyle1,
                   linewidth = fLineWidth1, label = sLabel1 )
        plt.plot ( aX, aY2, color = GetColor ( sColor2 ), marker = sMarker2, markersize = fMarkerSize2, linestyle = sLineStyle2,
                   linewidth = fLineWidth2, label = sLabel2 )
        plt.plot ( aX, aY3, color = GetColor ( sColor3 ), marker = sMarker3, markersize = fMarkerSize3, linestyle = sLineStyle3,
                   linewidth = fLineWidth3, label = sLabel3 )
        
    if ( ListAnnotation is not None ):
        DrawAnnotations ( ListAnnotation = ListAnnotation, GraphicConfig = GraphicConfig )
        
    if ( ListMarker is not None ):
        DrawMarker ( ListMarker = ListMarker, GraphicConfig = GraphicConfig, CAxis = None )
        
    if ( ( len ( sLabel1 ) > 0 ) or ( len ( sLabel2 ) > 0 ) or ( len ( sLabel3 ) > 0 ) ):
        plt.legend ( #[ sTextLegend, sTextLegend2, sTextLegend3 ], 
                     prop = { "family": GDictPlotParameter.get ( "FontName" ), "size": 12 }, loc = GraphicConfig.sLegendPosition )
    
    if ( ( ( len ( str ( aX[ 0 ] ) ) > 8 ) ) and ( len ( str ( aX[ -1 ] ) ) > 8 ) ):
        plt.setp ( plt.gca ().xaxis.get_majorticklabels (), "rotation", 50, "ha", "right" )
        
    if ( tRangeX ):
        plt.xlim ( tRangeX[ 0 ], tRangeX[ 1 ] )
    if ( tRangeY ):
        plt.ylim ( tRangeY[ 0 ], tRangeY[ 1 ] )
        
    if ( GraphicConfig.bShowPlot == True ):
        plt.show ()
    
    return

def Plot2X2Y ( aX1, aY1, aX2, aY2, tStyleY1, tStyleY2, GraphicConfig, tFillArea = None, ListMarker = None, ListAnnotation = None, 
               tRangeX = None, tRangeY = None ):
    plt.figure ( figsize = GraphicConfig.tFigureSize )
    sColor1, sMarker1, fMarkerSize1, sLineStyle1, fLineWidth1, sLabel1 = StyleCheck6 ( tStyleY1 )
    if ( len ( sLabel1 ) == 0 ):
        sLabel1 = GraphicConfig.sTextLegend
    sColor2, sMarker2, fMarkerSize2, sLineStyle2, fLineWidth2, sLabel2 = StyleCheck6 ( tStyleY2 )
    if ( len ( sLabel2 ) == 0 ):
        sLabel2 = GraphicConfig.sTextLegend2
        
    DrawFrame ( GraphicConfig = GraphicConfig, bShowLegend = False )
    
    if ( tFillArea is not None ): # tFillArea : ( obere Grenez pro X-Wert, untere Grtenze pro X-Wert, Farbcode, Alpha-Wert)
        CheckAssert ( bBool = ( isinstance ( tFillArea, ( tuple, list ) ) and ( len ( tFillArea ) == 5 ) ), 
                      sMsg = "Invalid Shape <tFillArea>!" )
        plt.fill_between ( aX1, tFillArea[ 0 ], tFillArea[ 1 ], facecolor = GetColor ( tFillArea[ 2 ] ), alpha = tFillArea[ 3 ],
                           label = tFillArea[ 4 ] )
    
    if ( GraphicConfig.sStepPlotWhere != "none" ):
        plt.step ( aX1, aY1, color = GetColor ( sColor1 ), marker = sMarker1, markersize = fMarkerSize1, linestyle = sLineStyle1,
                   where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth1, label = sLabel1 )
        plt.step ( aX2, aY2, color = GetColor ( sColor2 ), marker = sMarker2, markersize = fMarkerSize2, linestyle = sLineStyle2,
                   where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth2, label = sLabel2 )
    else:
        plt.plot ( aX1, aY1, color = GetColor ( sColor1 ), marker = sMarker1, markersize = fMarkerSize1, linestyle = sLineStyle1,
                   linewidth = fLineWidth1, label = sLabel1 )
        plt.plot ( aX2, aY2, color = GetColor ( sColor2 ), marker = sMarker2, markersize = fMarkerSize2, linestyle = sLineStyle2,
                   linewidth = fLineWidth2, label = sLabel2 )
             
    if ( ListAnnotation is not None ):
        DrawAnnotations ( ListAnnotation = ListAnnotation, GraphicConfig = GraphicConfig )
        
    if ( ListMarker is not None ):
        DrawMarker ( ListMarker = ListMarker, GraphicConfig = GraphicConfig, CAxis = None )
        
    if ( ( len ( sLabel1 ) > 0 ) or ( len ( sLabel2 ) > 0 ) ):
        plt.legend ( prop = { "family": GDictPlotParameter.get ( "FontName" ), "size": 12 }, loc = GraphicConfig.sLegendPosition )

    if ( ( ( len ( str ( aX1[ 0 ] ) ) > 8 ) ) and ( len ( str ( aX1[ -1 ] ) ) > 8 ) ):
        plt.setp ( plt.gca ().xaxis.get_majorticklabels (), "rotation", 50, "ha", "right" )
    
    if ( tRangeX ):
        plt.xlim ( tRangeX[ 0 ], tRangeX[ 1 ] )
    if ( tRangeY ):
        plt.ylim ( tRangeY[ 0 ], tRangeY[ 1 ] )
        
    plt.show ()
    
    return

def Plot2X3Y ( aX1, aY1, aY2, aX2, aY3, tStyleY1, tStyleY2, tStyleY3, GraphicConfig, tFillArea = None, ListMarker = None, ListAnnotation = None, 
               tRangeX = None, tRangeY = None ):   
    plt.figure ( figsize = GraphicConfig.tFigureSize )
    sColor1, sMarker1, fMarkerSize1, sLineStyle1, fLineWidth1, sLabel1 = StyleCheck6 ( tStyleY1 )
    if ( len ( sLabel1 ) == 0 ):
        sLabel1 = GraphicConfig.sTextLegend
    sColor2, sMarker2, fMarkerSize2, sLineStyle2, fLineWidth2, sLabel2 = StyleCheck6 ( tStyleY2 )
    if ( len ( sLabel2 ) == 0 ):
        sLabel2 = GraphicConfig.sTextLegend2
    sColor3, sMarker3, fMarkerSize3, sLineStyle3, fLineWidth3, sLabel3 = StyleCheck6 ( tStyleY3 )
    if ( len ( sLabel3 ) == 0 ):
        sLabel3 = GraphicConfig.sTextLegend3
        
    DrawFrame ( GraphicConfig = GraphicConfig, bShowLegend = False )
    
    if ( tFillArea is not None ): # tFillArea : ( obere Grenez pro X-Wert, untere Grtenze pro X-Wert, Farbcode, Alpha-Wert)
        CheckAssert ( bBool = ( isinstance ( tFillArea, ( tuple, list ) ) and ( len ( tFillArea ) == 5 ) ), 
                      sMsg = "Invalid Shape <tFillArea>!" )
        plt.fill_between ( aX1, tFillArea[ 0 ], tFillArea[ 1 ], facecolor = GetColor ( tFillArea[ 2 ] ), alpha = tFillArea[ 3 ],
                           label = tFillArea[ 4 ] )
    
    if ( GraphicConfig.sStepPlotWhere != "none" ):
        plt.step ( aX1, aY1, color = GetColor ( sColor1 ), marker = sMarker1, markersize = fMarkerSize1, linestyle = sLineStyle1,
                   where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth1, label = sLabel1 )
        plt.step ( aX1, aY2, color = GetColor ( sColor2 ), marker = sMarker2, markersize = fMarkerSize2, linestyle = sLineStyle2,
                   where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth2, label = sLabel2 )
        plt.step ( aX2, aY3, color = GetColor ( sColor3 ), marker = sMarker3, markersize = fMarkerSize3, linestyle = sLineStyle3,
                   where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth3, label = sLabel3 )
    else:
        plt.plot ( aX1, aY1, color = GetColor ( sColor1 ), marker = sMarker1, markersize = fMarkerSize1, linestyle = sLineStyle1,
                   linewidth = fLineWidth1, label = sLabel1 )
        plt.plot ( aX1, aY2, color = GetColor ( sColor2 ), marker = sMarker2, markersize = fMarkerSize2, linestyle = sLineStyle2,
                   linewidth = fLineWidth2, label = sLabel2 )
        plt.plot ( aX2, aY3, color = GetColor ( sColor3 ), marker = sMarker3, markersize = fMarkerSize3, linestyle = sLineStyle3,
                   linewidth = fLineWidth3, label = sLabel3 )
        
    if ( ListAnnotation is not None ):
        DrawAnnotations ( ListAnnotation = ListAnnotation, GraphicConfig = GraphicConfig )
        
    if ( ListMarker is not None ):
        DrawMarker ( ListMarker = ListMarker, GraphicConfig = GraphicConfig, CAxis = None )
            
    if ( ( len ( sLabel1 ) > 0 ) or ( len ( sLabel2 ) > 0 ) or ( len ( sLabel3 ) > 0 ) ):
        plt.legend ( #[ sTextLegend, sTextLegend2, sTextLegend3 ], 
                     prop = { "family": GDictPlotParameter.get ( "FontName" ), "size": 12 }, loc = GraphicConfig.sLegendPosition )
    
    if ( ( ( len ( str ( aX1[ 0 ] ) ) > 8 ) ) and ( len ( str ( aX1[ -1 ] ) ) > 8 ) ):
        plt.setp ( plt.gca ().xaxis.get_majorticklabels (), "rotation", 50, "ha", "right" )
        
    if ( tRangeX ):
        plt.xlim ( tRangeX[ 0 ], tRangeX[ 1 ] )
    if ( tRangeY ):
        plt.ylim ( tRangeY[ 0 ], tRangeY[ 1 ] )
        
    plt.show ()
    
    return

def Plot3X3Y ( aX1, aY1, aX2, aY2, aX3, aY3, tStyleY1, tStyleY2, tStyleY3, GraphicConfig, ListMarker = None, ListAnnotation = None, tRangeX = None, tRangeY = None ):   
    plt.figure ( figsize = GraphicConfig.tFigureSize )
    sColor1, sMarker1, fMarkerSize1, sLineStyle1, fLineWidth1, sLabel1 = StyleCheck6 ( tStyleY1 )
    if ( len ( sLabel1 ) == 0 ):
        sLabel1 = GraphicConfig.sTextLegend
    sColor2, sMarker2, fMarkerSize2, sLineStyle2, fLineWidth2, sLabel2 = StyleCheck6 ( tStyleY2 )
    if ( len ( sLabel2 ) == 0 ):
        sLabel2 = GraphicConfig.sTextLegend2
    sColor3, sMarker3, fMarkerSize3, sLineStyle3, fLineWidth3, sLabel3 = StyleCheck6 ( tStyleY3 )
    if ( len ( sLabel3 ) == 0 ):
        sLabel3 = GraphicConfig.sTextLegend3
        
    DrawFrame ( GraphicConfig = GraphicConfig, bShowLegend = False )
    
    if ( GraphicConfig.sStepPlotWhere != "none" ):
        plt.step ( aX1, aY1, color = GetColor ( sColor1 ), marker = sMarker1, markersize = fMarkerSize1, linestyle = sLineStyle1,
                   where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth1, label = sLabel1 )
        plt.step ( aX2, aY2, color = GetColor ( sColor2 ), marker = sMarker2, markersize = fMarkerSize2, linestyle = sLineStyle2,
                   where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth2, label = sLabel2 )
        plt.step ( aX3, aY3, color = GetColor ( sColor3 ), marker = sMarker3, markersize = fMarkerSize3, linestyle = sLineStyle3,
                   where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth3, label = sLabel3 )
    else:
        plt.plot ( aX1, aY1, color = GetColor ( sColor1 ), marker = sMarker1, markersize = fMarkerSize1, linestyle = sLineStyle1,
                   linewidth = fLineWidth1, label = sLabel1 )
        plt.plot ( aX2, aY2, color = GetColor ( sColor2 ), marker = sMarker2, markersize = fMarkerSize2, linestyle = sLineStyle2,
                   linewidth = fLineWidth2, label = sLabel2 )
        plt.plot ( aX3, aY3, color = GetColor ( sColor3 ), marker = sMarker3, markersize = fMarkerSize3, linestyle = sLineStyle3,
                   linewidth = fLineWidth3, label = sLabel3 )
        
    if ( ListAnnotation is not None ):
        DrawAnnotations ( ListAnnotation = ListAnnotation, GraphicConfig = GraphicConfig )
        
    if ( ListMarker is not None ):
        DrawMarker ( ListMarker = ListMarker, GraphicConfig = GraphicConfig, CAxis = None )
            
    if ( ( len ( sLabel1 ) > 0 ) or ( len ( sLabel2 ) > 0 ) or ( len ( sLabel3 ) > 0 ) ):
        plt.legend ( #[ sTextLegend, sTextLegend2, sTextLegend3 ], 
                        prop = { "family": GDictPlotParameter.get ( "FontName" ), "size": 12 }, loc = GraphicConfig.sLegendPosition )
    
    if ( ( ( len ( str ( aX1[ 0 ] ) ) > 8 ) ) and ( len ( str ( aX1[ -1 ] ) ) > 8 ) ):
        plt.setp ( plt.gca ().xaxis.get_majorticklabels (), "rotation", 50, "ha", "right" )
        
    if ( tRangeX ):
        plt.xlim ( tRangeX[ 0 ], tRangeX[ 1 ] )
    if ( tRangeY ):
        plt.ylim ( tRangeY[ 0 ], tRangeY[ 1 ] )
        
    plt.show ()
    
    return

def PlotX4Y ( aX, tY, tStylesY, GraphicConfig, tFillArea = None, ListMarker = None, ListAnnotation = None, tRangeX = None, tRangeY = None ):   
    CheckAssert ( bBool = ( ( len ( tY ) == len ( tStylesY ) ) and ( len ( tY ) == 4 ) ), sMsg = "Invalid Shape <tStylesY> and <tY>!" )
    aY1, aY2, aY3, aY4 = tY
    plt.figure ( figsize = GraphicConfig.tFigureSize )
    
    sColor1, sMarker1, fMarkerSize1, sLineStyle1, fLineWidth1, sLabel1 = StyleCheck6 ( tStylesY[ 0 ] )
    if ( len ( sLabel1 ) == 0 ):
        sLabel1 = GraphicConfig.sTextLegend
    sColor2, sMarker2, fMarkerSize2, sLineStyle2, fLineWidth2, sLabel2 = StyleCheck6 ( tStylesY[ 1 ] )
    if ( len ( sLabel2 ) == 0 ):
        sLabel2 = GraphicConfig.sTextLegend2
    sColor3, sMarker3, fMarkerSize3, sLineStyle3, fLineWidth3, sLabel3 = StyleCheck6 ( tStylesY[ 2 ] )
    if ( len ( sLabel3 ) == 0 ):
        sLabel3 = GraphicConfig.sTextLegend3
    sColor4, sMarker4, fMarkerSize4, sLineStyle4, fLineWidth4, sLabel4 = StyleCheck6 ( tStylesY[ 3 ] )
    if ( len ( sLabel4 ) == 0 ):
        sLabel4 = GraphicConfig.sTextLegend4
        
    DrawFrame ( GraphicConfig = GraphicConfig, bShowLegend = False )
    
    if ( tFillArea is not None ): # tFillArea : ( obere Grenez pro X-Wert, untere Grtenze pro X-Wert, Farbcode, Alpha-Wert)
        CheckAssert ( bBool = ( isinstance ( tFillArea, ( tuple, list ) ) and ( len ( tFillArea ) == 5 ) ), 
                      sMsg = "Invalid Shape <tFillArea>!" )
        plt.fill_between ( aX, tFillArea[ 0 ], tFillArea[ 1 ], facecolor = GetColor ( tFillArea[ 2 ] ), alpha = tFillArea[ 3 ],
                           label = tFillArea[ 4 ] )
    
    if ( GraphicConfig.sStepPlotWhere != "none" ):
        plt.step ( aX, aY1, color = GetColor ( sColor1 ), marker = sMarker1, markersize = fMarkerSize1, linestyle = sLineStyle1,
                   where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth1, label = sLabel1 )
        plt.step ( aX, aY2, color = GetColor ( sColor2 ), marker = sMarker2, markersize = fMarkerSize2, linestyle = sLineStyle2,
                   where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth2, label = sLabel2 )
        plt.step ( aX, aY3, color = GetColor ( sColor3 ), marker = sMarker3, markersize = fMarkerSize3, linestyle = sLineStyle3,
                   where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth3, label = sLabel3 )
        plt.step ( aX, aY4, color = GetColor ( sColor4 ), marker = sMarker4, markersize = fMarkerSize4, linestyle = sLineStyle4,
                   where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth4, label = sLabel4 )
    else:
        plt.plot ( aX, aY1, color = GetColor ( sColor1 ), marker = sMarker1, markersize = fMarkerSize1, linestyle = sLineStyle1,
                   linewidth = fLineWidth1, label = sLabel1 )
        plt.plot ( aX, aY2, color = GetColor ( sColor2 ), marker = sMarker2, markersize = fMarkerSize2, linestyle = sLineStyle2,
                   linewidth = fLineWidth2, label = sLabel2 )
        plt.plot ( aX, aY3, color = GetColor ( sColor3 ), marker = sMarker3, markersize = fMarkerSize3, linestyle = sLineStyle3,
                   linewidth = fLineWidth3, label = sLabel3 )
        plt.plot ( aX, aY4, color = GetColor ( sColor4 ), marker = sMarker4, markersize = fMarkerSize4, linestyle = sLineStyle4,
                   linewidth = fLineWidth4, label = sLabel4 )
        
    if ( ListAnnotation is not None ):
        DrawAnnotations ( ListAnnotation = ListAnnotation, GraphicConfig = GraphicConfig )
        
    if ( ListMarker is not None ):
        DrawMarker ( ListMarker = ListMarker, GraphicConfig = GraphicConfig, CAxis = None )
        
    if ( ( len ( sLabel1 ) > 0 ) or ( len ( sLabel2 ) > 0 ) or ( len ( sLabel3 ) > 0 ) or ( len ( sLabel4 ) > 0 )):
        plt.legend ( #[ sTextLegend, sTextLegend2, sTextLegend3 ], 
                     prop = { "family": GDictPlotParameter.get ( "FontName" ), "size": 12 }, loc = GraphicConfig.sLegendPosition )
    
    if ( ( ( len ( str ( aX[ 0 ] ) ) > 8 ) ) and ( len ( str ( aX[ -1 ] ) ) > 8 ) ):
        plt.setp ( plt.gca ().xaxis.get_majorticklabels (), "rotation", 50, "ha", "right" )
        
    if ( tRangeX ):
        plt.xlim ( tRangeX[ 0 ], tRangeX[ 1 ] )
    if ( tRangeY ):
        plt.ylim ( tRangeY[ 0 ], tRangeY[ 1 ] )
        
    if ( GraphicConfig.bShowPlot == True ):
        plt.show ()
    
    return
# ************************** Visualisierung von Bildern mit Titel, sowie X- und Y-Beschriftung und ggf. Farb-Code **************************
#                               Colorbar nur dann, wenn die Beschriftung via sTextLegend aktiviert wird
#                                   tExtent = ( left, right, bottom, top )
def PlotImage ( aData2Dim, GraphicConfig, sColorMap, sInterpolation = "spline36", sOrigin = "upper", tExtent = None, sGridAxis = "" ):
    ## Nur eine Auswahl der möglichen Interpolations-Schemata
    CheckAssert ( bBool = ( sInterpolation in  [ None, "none", "nearest", "bilinear", "spline16", "spline36", "hanning", "hermite", 
                                                 "gaussian", "bessel", "sinc", "lanczos" ] ), sMsg = "Invalid Interpolation Scheme!" )
    if ( tExtent is not None ):
        CheckAssert ( bBool = ( len ( tExtent ) == 4 ), sMsg = "Wrong Format <tExtent>!" )
    CFigure, CAxis = plt.subplots ( figsize = GraphicConfig.tFigureSize )
    CImg = CAxis.imshow ( aData2Dim, cmap = sColorMap, origin = sOrigin, interpolation = sInterpolation, interpolation_stage = "data",
                          extent = tExtent )
    if ( GraphicConfig.sTextLegend ):
        CColorBar = CAxis.figure.colorbar ( CImg, ax = CAxis )
        CColorBar.ax.set_ylabel ( GraphicConfig.sTextLegend, fontname = GDictPlotParameter.get ( "FontName" ), 
                                  fontsize = GDictPlotParameter.get ( "LabelSizeColorbar" ), rotation = -90, verticalalignment = "bottom" )
    
        for CLabel in CColorBar.ax.get_yticklabels ():
            CLabel.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
            CLabel.set_fontsize ( fontsize = GDictPlotParameter.get ( "TickSizeColorbar" ) )
        
    DrawFrameAxis ( CAxis = CAxis, GraphicConfig = GraphicConfig, iIndex = 0, tSizes = None, sGridAxis = sGridAxis )
    plt.show ()

    return    

## ________________________________________________________ Gestapelte Grafiken ____________________________________________________________ 
## -----------------------------------------------------------------------------------------------------------------------------------------

# ******************************************* Vertikal gestapelte Plots (X1, Y1) und (X2, Y2) **********************************************
def PlotStackVert2X2Y ( aX1, aX2, aY1, aY2, GraphicConfig, tStyle1, tStyle2, ListAnnotations = None, ListMarkers = None, bShareX = False ):
    CFigure, tCAxes = plt.subplots ( nrows = 2, sharey = False, sharex = bShareX, 
                                     figsize = GDictPlotParameter.get ( "FigureSizeStackVert" ) )
    
    ListLegends = GraphicConfig.GetTextLegends () 
  
    tStyles = ( tStyle1, tStyle2 )
    tX = ( aX1, aX2 )
    tData = ( aY1, aY2 )
    ## falls über tStyle ein Label übergeben wird, so wird dieses übernommen
    for ik in range ( 2 ):
        if ( tStyles[ ik ][ 5 ] ): 
            ListLegends[ ik ] = tStyles[ ik ][ 5 ]
                 
    for ik in range ( len ( tCAxes ) ):
        DrawFrameAxis ( CAxis = tCAxes[ ik ], GraphicConfig = GraphicConfig, iIndex = ik, sGridAxis = GraphicConfig.sGridAxis, tSizes = ( 20, 18, 16 ) )
                
        sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, _ = StyleCheck6 ( tStyles[ ik ] ) 
        
        if ( GraphicConfig.sStepPlotWhere != "none" ):
            tCAxes[ ik ].step ( tX[ ik ], tData[ ik ], marker = sMarker, ms = fMarkerSize, color = GetColor ( sColor ), 
                                linestyle = sLineStyle, where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth, label = ListLegends[ ik ] )
        else:
            tCAxes[ ik ].plot ( tX[ ik ], tData[ ik ], marker = sMarker, ms = fMarkerSize, color = GetColor ( sColor ), 
                                linestyle = sLineStyle, linewidth = fLineWidth, label = ListLegends[ ik ] )
        
        if ( ListAnnotations is not None and len ( ListAnnotations ) >= ik ): ### um nur eine Liste zu übergeben: ListAnnotations = [ Liste1, None ]
            ListAnnotation = ListAnnotations[ ik ]
            if ( ListAnnotation is not None ):
                DrawAnnotations ( ListAnnotation = ListAnnotation, GraphicConfig = GraphicConfig, CAxis = tCAxes[ ik ] )
        
        if ( ListMarkers is not None and len ( ListMarkers ) >= ik ): ### um nur eine Liste zu übergeben: ListMarkers = [ Liste1, None ]
            ListMarker = ListMarkers[ ik ]
            if ( ListMarker is not None ):
                DrawMarker ( ListMarker = ListMarker, GraphicConfig = GraphicConfig, CAxis = tCAxes[ ik ] )
        
        if ( ListLegends[ ik ] ):
            tCAxes[ ik ].legend ( prop = { "family": GDictPlotParameter.get ( "FontName" ), "size": GDictPlotParameter.get ( "LegendSize" ) }, 
                                 loc = GraphicConfig.sLegendPosition )
                   
    plt.tight_layout ()
    plt.show ()
    
    return
    
def PlotStackVertX2Y ( aX, aY1, aY2, GraphicConfig, tStyle1, tStyle2, ListAnnotations = None, ListMarkers = None, bShareX = False ):
    PlotStackVert2X2Y ( aX1 = aX, aX2 = aX, aY1 = aY1, aY2 = aY2, GraphicConfig = GraphicConfig, tStyle1 = tStyle1, tStyle2 = tStyle2, 
                        ListAnnotations = ListAnnotations, ListMarkers = ListMarkers, bShareX = bShareX )
    
    return

def PlotQuartett ( uX, tY, GraphicConfig, tStyles ):
    if ( isinstance ( uX, ( np.ndarray ) ) ):
        tX = ( uX, uX, uX, uX )
    else:
        tX = uX
    
    CheckAssert ( bBool = ( ( len ( tX ) == 4 ) and ( len ( tY ) == 4 ) and ( len ( tStyles ) == 4 ) ), sMsg = "Shape Mismatch!" )
    
    ListLegends = GraphicConfig.GetTextLegends () 
    #tSizes = iTitleSize, iLabelSize, iTickSize 
    tSizes = ( 22, 18, 14 )
    ## falls über tStyle ein Label übergeben wird, so wird dieses übernommen
    for ik in range ( len ( ListLegends ) ):
        if ( tStyles[ ik ][ 5 ] ): 
            ListLegends[ ik ] = tStyles[ ik ][ 5 ]
    
    CFigure, tCAxes = plt.subplots ( nrows = 2, ncols = 2, sharey = False, sharex = False, 
                                     figsize = GDictPlotParameter.get ( "FigureSizeStackVert2x2" ) )
    
    for ik, CAxis in enumerate ( np.ravel ( tCAxes ) ):
        DrawFrameAxis ( CAxis = CAxis, GraphicConfig = GraphicConfig, iIndex = ik, sGridAxis = GraphicConfig.sGridAxis, tSizes = tSizes )
        
        sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, _ = StyleCheck6 ( tStyles[ ik ] ) 
        CAxis.plot ( tX[ ik ], tY[ ik ], marker = sMarker, ms = fMarkerSize, color = GetColor ( sColor ), 
                     linestyle = sLineStyle, linewidth = fLineWidth, label = ListLegends[ ik ] )
        
        if ( ListLegends[ ik ] ):
            CAxis.legend ( prop = { "family": GDictPlotParameter.get ( "FontName" ), "size": GDictPlotParameter.get ( "LegendSize" ) }, 
                           loc = GraphicConfig.sLegendPosition )
            
        if ( ( ( len ( str ( uX[ 0 ] ) ) > 8 ) ) and ( len ( str ( uX[ -1 ] ) ) > 8 ) ):
            CAxis.tick_params ( axis = "x", which = "major", labelrotation = 45 )
    
    #if ( ( ( len ( str ( uX[ 0 ] ) ) > 8 ) ) and ( len ( str ( uX[ -1 ] ) ) > 8 ) ):
    #    plt.setp ( plt.gca ().xaxis.get_majorticklabels (), "rotation", 50, "ha", "right" )
        
    plt.tight_layout ()
    plt.show ()
    
    return

# *********************************** Horizontal gestapelte (nebeneinander) Plots (X1, Y1) und (X2, Y2) ************************************
def PlotStackHorz2X2Y ( aX1, aY1, aX2, aY2, GraphicConfig, tStyles, ListAnnotations = None, bShareY = True ):
    CheckAssert ( bBool = ( ( aX1.shape[ 0 ] == aY1.shape[ 0 ] ) and ( aX2.shape[ 0 ] == aY2.shape[ 0 ] ) and 
                            ( len ( tStyles ) == 2 ) ), sMsg = "Invalid Shape of Input!" )
    
    CFigure, tCAxes = plt.subplots ( ncols = 2, sharey = bShareY, figsize = GDictPlotParameter.get ( "FigureSizeStack2Horz" ) )
    tX = ( aX1, aX2 )
    tData = ( aY1, aY2 )
    ListLegends = GraphicConfig.GetTextLegends () 
    
    ## falls über tStyle ein Label übergeben wird, so wird dieses übernommen
    for ik in range ( 2 ):
        if ( tStyles[ ik ][ 5 ] ): 
            ListLegends[ ik ] = tStyles[ ik ][ 5 ]
                 
    for ik in range ( len ( tCAxes ) ):
        DrawFrameAxis ( CAxis = tCAxes[ ik ], GraphicConfig = GraphicConfig, iIndex = ik, sGridAxis = GraphicConfig.sGridAxis )
        sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, _ = StyleCheck6 ( tStyles[ ik ] ) 
        if ( GraphicConfig.sStepPlotWhere != "none" ):
            tCAxes[ ik ].step ( tX[ ik ], tData[ ik ], marker = sMarker, ms = fMarkerSize, color = GetColor ( sColor ), 
                                linestyle = sLineStyle, where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth, label = ListLegends[ ik ] )
        else:
            tCAxes[ ik ].plot ( tX[ ik ], tData[ ik ], marker = sMarker, ms = fMarkerSize, color = GetColor ( sColor ), 
                                linestyle = sLineStyle, linewidth = fLineWidth, label = ListLegends[ ik ] )
            
        if ( ListAnnotations is not None and len ( ListAnnotations ) >= ik ):
            ListAnnotation = ListAnnotations[ ik ]
            if ( ListAnnotation is not None ):
                DrawAnnotations ( ListAnnotation = ListAnnotation, GraphicConfig = GraphicConfig, CAxis = tCAxes[ ik ] )
                
        if ( ListLegends[ ik ] ):
            tCAxes[ ik ].legend ( prop = { "family": GDictPlotParameter.get ( "FontName" ), "size": GDictPlotParameter.get ( "LegendSize" ) }, 
                                 loc = GraphicConfig.sLegendPosition )
                   
    plt.tight_layout ()
    plt.show ()
    
    return
# *********************************** Horizontal gestapelte (nebeneinander) Plots (X1, Y1) und (X2, Y2) ************************************
def PlotStackHorz3X3Y ( aX1, aY1, aX2, aY2, aX3, aY3, GraphicConfig, tStyles, ListAnnotations = None, bShareY = True ):
    CheckAssert ( bBool = ( ( aX1.shape[ 0 ] == aY1.shape[ 0 ] ) and ( aX2.shape[ 0 ] == aY2.shape[ 0 ] ) and 
                            ( aX3.shape[ 0 ] == aY3.shape[ 0 ] ) and ( len ( tStyles ) == 3 ) ), sMsg = "Invalid Shape of Input!" )
    
    CFigure, tCAxes = plt.subplots ( ncols = 3, sharey = bShareY, figsize = GDictPlotParameter.get ( "FigureSizeStack3Horz" ) )
    tX = ( aX1, aX2, aX3 )
    tData = ( aY1, aY2, aY3 )
    ListLegends = GraphicConfig.GetTextLegends () 
    
    ## falls über tStyle ein Label übergeben wird, so wird dieses übernommen
    for ik in range ( 3 ):
        if ( tStyles[ ik ][ 5 ] ): 
            ListLegends[ ik ] = tStyles[ ik ][ 5 ]
                 
    for ik in range ( len ( tCAxes ) ):
        DrawFrameAxis ( CAxis = tCAxes[ ik ], GraphicConfig = GraphicConfig, iIndex = ik, sGridAxis = GraphicConfig.sGridAxis )
        sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, _ = StyleCheck6 ( tStyles[ ik ] ) 
        if ( GraphicConfig.sStepPlotWhere != "none" ):
            tCAxes[ ik ].step ( tX[ ik ], tData[ ik ], marker = sMarker, ms = fMarkerSize, color = GetColor ( sColor ), 
                                linestyle = sLineStyle, where = GraphicConfig.sStepPlotWhere, linewidth = fLineWidth, label = ListLegends[ ik ] )
        else:
            tCAxes[ ik ].plot ( tX[ ik ], tData[ ik ], marker = sMarker, ms = fMarkerSize, color = GetColor ( sColor ), 
                                linestyle = sLineStyle, linewidth = fLineWidth, label = ListLegends[ ik ] )
            
        if ( ListAnnotations is not None and len ( ListAnnotations ) >= ik ):
            ListAnnotation = ListAnnotations[ ik ]
            if ( ListAnnotation is not None ):
                DrawAnnotations ( ListAnnotation = ListAnnotation, GraphicConfig = GraphicConfig, CAxis = tCAxes[ ik ] )
        
        if ( ListLegends[ ik ] ):
            tCAxes[ ik ].legend ( prop = { "family": GDictPlotParameter.get ( "FontName" ), "size": GDictPlotParameter.get ( "LegendSize" ) }, 
                                 loc = GraphicConfig.sLegendPosition )
                   
    plt.tight_layout ()
    plt.show ()
    
    return
# ********************************** Plot 2-dim Daten und daneben eine normale X-Y Grafik **************************************************
def PlotStackHorzImageXY ( aX, aData, aData2Dim, tStyle, GraphicConfig, sColorMap = "coolwarm", tExtent = None, sOrigin = "upper" ):
    CheckAssert ( bBool = ( ( aX.shape[ 0 ] == aData.shape[ 0 ] ) and ( aData2Dim.ndim == 2 ) ), sMsg = "Invalid Shape of Input!" )
    
    CFigure, tCAxes = plt.subplots ( ncols = 2, figsize = GDictPlotParameter.get ( "FigureSizeStackHorz" ), layout = "constrained" )
    ListLegends = GraphicConfig.GetTextLegends () 
     
    ## falls über tStyle ein Label übergeben wird, so wird dieses übernommen
    if ( tStyle[ 5 ] ): 
        ListLegends[ 1 ] = tStyle[ 5 ]
          
    for ik in range ( len ( tCAxes ) ):
        DrawFrameAxis ( CAxis = tCAxes[ ik ], GraphicConfig = GraphicConfig, iIndex = ik, sGridAxis = GraphicConfig.sGridAxis )
    
    ## Bild nach links
    tCAxes[ 0 ].imshow ( aData2Dim, cmap = sColorMap, origin = sOrigin, extent = tExtent )
    
    ## XY-Plot nach rechts
    sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, _ = StyleCheck6 ( tStyle ) 
    tCAxes[ 1 ].set_box_aspect ( aData2Dim.shape[ 0 ] / aData2Dim.shape[ 1 ] )
    tCAxes[ 1 ].plot ( aX, aData, marker = sMarker, ms = fMarkerSize, color = GetColor ( sColor ), 
                       linestyle = sLineStyle, linewidth = fLineWidth, label = ListLegends[ 1 ] )

    if ( ListLegends[ 1 ] ):
        tCAxes[ 1 ].legend ( prop = { "family": GDictPlotParameter.get ( "FontName" ), "size": GDictPlotParameter.get ( "LegendSize" ) }, 
                             loc = GraphicConfig.sLegendPosition )
            
    if ( GraphicConfig.sGridAxis in [ "x", "y", "both" ] ):
        tCAxes[ 1 ].grid ( visible = True, axis = GraphicConfig.sGridAxis )
    else:
        tCAxes[ 1 ].grid ( visible = False )
                
    plt.tight_layout ()
    plt.show ()
        
    return

# ********************************** Plot 2-dim Daten und daneben eine normale X-2Y Grafik *************************************************
def PlotStackHorzImageX2Y ( aX, aData, aData2Dim, tStyleY1, tStyleY2, GraphicConfig, sColorMap = "coolwarm", tExtent = None, sOrigin = "upper" ):
    CheckAssert ( bBool = ( ( aX.shape[ 0 ] == aData.shape[ 0 ] ) and ( aData.shape[ 1 ] == 2 ) and ( aData2Dim.ndim == 2 ) ), 
                  sMsg = "Invalid Shape of Input!" )
    
    CFigure, tCAxes = plt.subplots ( ncols = 2, figsize = GDictPlotParameter.get ( "FigureSizeStackHorz" ), layout = "constrained" )
    ListLegends = GraphicConfig.GetTextLegends () 
     
    ## falls über tStyle ein Label übergeben wird, so wird dieses übernommen
    if ( tStyleY1[ 5 ] ): 
        ListLegends[ 0 ] = tStyleY1[ 5 ]
    if ( tStyleY2[ 5 ] ): 
        ListLegends[ 1 ] = tStyleY2[ 5 ]
          
    for ik in range ( len ( tCAxes ) ):
        DrawFrameAxis ( CAxis = tCAxes[ ik ], GraphicConfig = GraphicConfig, iIndex = ik, sGridAxis = GraphicConfig.sGridAxis )
    
    ## Bild nach links
    tCAxes[ 0 ].imshow ( aData2Dim, cmap = sColorMap, origin = sOrigin, extent = tExtent )
    
    ## XY-Plot nach rechts
    sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, _ = StyleCheck6 ( tStyleY1 ) 
    tCAxes[ 1 ].set_box_aspect ( aData2Dim.shape[ 0 ] / aData2Dim.shape[ 1 ] )
    tCAxes[ 1 ].plot ( aX, aData[ :, 0 ], marker = sMarker, ms = fMarkerSize, color = GetColor ( sColor ), 
                       linestyle = sLineStyle, linewidth = fLineWidth, label = ListLegends[ 0 ] )
    
    sColor, sMarker, fMarkerSize, sLineStyle, fLineWidth, _ = StyleCheck6 ( tStyleY2 ) 
    tCAxes[ 1 ].plot ( aX, aData[ :, 1 ], marker = sMarker, ms = fMarkerSize, color = GetColor ( sColor ), 
                       linestyle = sLineStyle, linewidth = fLineWidth, label = ListLegends[ 1 ] )

    if ( ListLegends[ 0 ] or ListLegends[ 1 ] ):
        tCAxes[ 1 ].legend ( prop = { "family": GDictPlotParameter.get ( "FontName" ), "size": GDictPlotParameter.get ( "LegendSize" ) }, 
                             loc = GraphicConfig.sLegendPosition )
            
    if ( GraphicConfig.sGridAxis in [ "x", "y", "both" ] ):
        tCAxes[ 1 ].grid ( visible = True, axis = GraphicConfig.sGridAxis )
    else:
        tCAxes[ 1 ].grid ( visible = False )
                
    plt.tight_layout ()
    plt.show ()
        
    return
## _____________________________________________________ ENDE : Gestapelte Grafiken ________________________________________________________ 
## -----------------------------------------------------------------------------------------------------------------------------------------

# ***************************************** Darstellung von 2*N Bildern in N Zeilen mit 2 Spalten ******************************************
def PlotImagesNx2 ( tImageArray, tTitle = None, sTickOption = "left", sColorMap = None, sInterpolation = "spline36", sOrigin = "upper" ):
    CheckAssert ( bBool = ( len ( tImageArray ) in [ 2, 4 ] ), sMsg = "Parameter Shape Mismatch!" )
    if ( tTitle is not None ):
        CheckAssert ( bBool = ( len ( tImageArray ) == len ( tTitle ) ), sMsg = "Parameter Mismatch <tTitle> and <tImageArray>!" )
    
    sTickOption = sTickOption.lower ()
    CheckAssert ( bBool = ( sTickOption in [ "left", "all", "", "none" ] ), sMsg = "Invalid Parameter <sTickOption>!" )
    
    if ( len ( tImageArray ) == 2 ):
        CFigure = plt.figure ( figsize = GDictPlotParameter.get ( "FigureSizeStackHorz" ) )
        gs = CFigure.add_gridspec ( 1, 2, hspace = -0.55, wspace = 0.12 )
    elif ( len ( tImageArray ) == 6 ):
        CFigure = plt.figure ( figsize = GDictPlotParameter.get ( "FigureSizeStackVert2x2" ) )
        gs = CFigure.add_gridspec ( 2, 2, hspace = -0.45, wspace = 0.12 )

    aCAxes = gs.subplots ()
    
    if ( aCAxes.ndim == 1 ):
        aCAxes = np.reshape ( aCAxes, shape = ( 1, aCAxes.shape[ 0 ] ) )

    for ik, CAxis in enumerate ( aCAxes ):
        CAxis[ ik % 2 ].imshow ( tImageArray[ 3 * ik ], cmap = sColorMap, origin = sOrigin, interpolation = sInterpolation, 
                                 interpolation_stage = "data" )
        if ( tTitle is not None ):
            CAxis[ ik % 2 ].set_title ( tTitle[ ik % 3 ], fontname = GDictPlotParameter.get ( "FontName" ), 
                                        fontsize = GDictPlotParameter.get ( "TitleSize2HorzStackImages" ) )
        if ( ( sTickOption == "left" ) or ( sTickOption == "all" ) ):
            for CLabel in CAxis[ ik % 3 ].get_xticklabels ():
                CLabel.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
                CLabel.set_fontsize ( fontsize = GDictPlotParameter.get ("TickSize2HorzStackImages" ) )
            for CLabel in CAxis[ ik % 3 ].get_yticklabels ():
                CLabel.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
                CLabel.set_fontsize ( fontsize = GDictPlotParameter.get ("TickSize2HorzStackImages" ) )
        else:
            CAxis[ ik % 3 ].get_xaxis ().set_visible ( False )
            CAxis[ ik % 3 ].get_yaxis ().set_visible ( False )
        
        CAxis[ ( ik + 1 ) % 3 ].imshow ( tImageArray[ 3 * ik + 1 ], cmap = sColorMap, origin = sOrigin, interpolation = sInterpolation, 
                                         interpolation_stage = "data" )
        if ( tTitle is not None ):
            CAxis[ ( ik + 1 ) % 2 ].set_title ( tTitle[ ( ik + 1 ) % 3 ], fontname = GDictPlotParameter.get ( "FontName" ), 
                                                fontsize = GDictPlotParameter.get ( "TitleSize2HorzStackImages" ) )
        if ( sTickOption == "all" ):
            for CLabel in CAxis[ ( ik + 1 ) % 3 ].get_xticklabels ():
                CLabel.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
                CLabel.set_fontsize ( fontsize = GDictPlotParameter.get ("TickSize2HorzStackImages" ) )
            for CLabel in CAxis[ ( ik + 1 ) % 3 ].get_yticklabels ():
                CLabel.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
                CLabel.set_fontsize ( fontsize = GDictPlotParameter.get ("TickSize2HorzStackImages" ) )
        else:
            CAxis[ ( ik + 1 ) % 2 ].get_xaxis ().set_visible ( False )
            CAxis[ ( ik + 1 ) % 2 ].get_yaxis ().set_visible ( False )
        
    plt.show ()
    
    return
# ***************************************** Darstellung von 3*N Bildern in N Zeilen mit 3 Spalten ******************************************
def PlotImagesNx3 ( tImageArray, tTitle = None, sTickOption = "left", sColorMap = None, sInterpolation = "spline36", sOrigin = "upper" ):
    CheckAssert ( bBool = ( len ( tImageArray ) in [ 3, 6 ] ), sMsg = "Parameter Shape Mismatch!" )
    if ( tTitle is not None ):
        CheckAssert ( bBool = ( len ( tImageArray ) == len ( tTitle ) ), sMsg = "Parameter Mismatch <tTitle> and <tImageArray>!" )
    
    sTickOption = sTickOption.lower ()
    CheckAssert ( bBool = ( sTickOption in [ "left", "all", "", "none" ] ), sMsg = "Invalid Parameter <sTickOption>!" )
    
    if ( len ( tImageArray ) == 3 ):
        CFigure = plt.figure ( figsize = GDictPlotParameter.get ( "FigureSizeStackHorz" ) )
        gs = CFigure.add_gridspec ( 1, 3, hspace = -0.55, wspace = 0.12 )
    elif ( len ( tImageArray ) == 6 ):
        CFigure = plt.figure ( figsize = ( 13, 14 ) )
        gs = CFigure.add_gridspec ( 2, 3, hspace = -0.45, wspace = 0.12 )

    aCAxes = gs.subplots ()
    
    if ( aCAxes.ndim == 1 ):
        aCAxes = np.reshape ( aCAxes, shape = ( 1, aCAxes.shape[ 0 ] ) )

    for ik, CAxis in enumerate ( aCAxes ):
        CAxis[ ik % 3 ].imshow ( tImageArray[ 3 * ik ], cmap = sColorMap, origin = sOrigin, interpolation = sInterpolation, 
                                 interpolation_stage = "data" )
        if ( tTitle is not None ):
            CAxis[ ik % 3 ].set_title ( tTitle[ ik % 3 ], fontname = GDictPlotParameter.get ( "FontName" ), 
                                        fontsize = GDictPlotParameter.get ( "TitleSize3HorzStackImages" ) )
        if ( ( sTickOption == "left" ) or ( sTickOption == "all" ) ):
            for CLabel in CAxis[ ik % 3 ].get_xticklabels ():
                CLabel.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
                CLabel.set_fontsize ( fontsize = GDictPlotParameter.get ("TickSize3HorzStackImages" ) )
            for CLabel in CAxis[ ik % 3 ].get_yticklabels ():
                CLabel.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
                CLabel.set_fontsize ( fontsize = GDictPlotParameter.get ("TickSize3HorzStackImages" ) )
        else:
            CAxis[ ik % 3 ].get_xaxis ().set_visible ( False )
            CAxis[ ik % 3 ].get_yaxis ().set_visible ( False )
        
        CAxis[ ( ik + 1 ) % 3 ].imshow ( tImageArray[ 3 * ik + 1 ], cmap = sColorMap, origin = sOrigin, interpolation = sInterpolation, 
                                         interpolation_stage = "data" )
        if ( tTitle is not None ):
            CAxis[ ( ik + 1 ) % 3 ].set_title ( tTitle[ ( ik + 1 ) % 3 ], fontname = GDictPlotParameter.get ( "FontName" ), 
                                                fontsize = GDictPlotParameter.get ( "TitleSize3HorzStackImages" ) )
        if ( sTickOption == "all" ):
            for CLabel in CAxis[ ( ik + 1 ) % 3 ].get_xticklabels ():
                CLabel.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
                CLabel.set_fontsize ( fontsize = GDictPlotParameter.get ("TickSize3HorzStackImages" ) )
            for CLabel in CAxis[ ( ik + 1 ) % 3 ].get_yticklabels ():
                CLabel.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
                CLabel.set_fontsize ( fontsize = GDictPlotParameter.get ("TickSize3HorzStackImages" ) )
        else:
            CAxis[ ( ik + 1 ) % 3 ].get_xaxis ().set_visible ( False )
            CAxis[ ( ik + 1 ) % 3 ].get_yaxis ().set_visible ( False )
        
        CAxis[ ( ik + 2 ) % 3 ].imshow ( tImageArray[ 3 * ik + 2 ], cmap = sColorMap, origin = sOrigin, interpolation = sInterpolation, 
                                         interpolation_stage = "data" )
        if ( tTitle is not None ):
            CAxis[ ( ik + 2 ) % 3 ].set_title ( tTitle[ ( ik + 2 ) % 3 ], fontname = GDictPlotParameter.get ( "FontName" ), 
                                                fontsize = GDictPlotParameter.get ( "TitleSize3HorzStackImages" ) )
        if ( sTickOption == "all" ):
            for CLabel in CAxis[ ( ik + 2 ) % 3 ].get_xticklabels ():
                CLabel.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
                CLabel.set_fontsize ( fontsize = GDictPlotParameter.get ("TickSize3HorzStackImages" ) )
            for CLabel in CAxis[ ( ik + 2 ) % 3 ].get_yticklabels ():
                CLabel.set_fontname ( fontname = GDictPlotParameter.get ( "FontName" ) )
                CLabel.set_fontsize ( fontsize = GDictPlotParameter.get ("TickSize3HorzStackImages" ) )
        else:
            CAxis[ ( ik + 2 ) % 3 ].get_xaxis ().set_visible ( False )
            CAxis[ ( ik + 2 ) % 3 ].get_yaxis ().set_visible ( False )

    plt.show ()
    
    return
# ****************************************************** Spaltet einen String in Text und Zahl *********************************************
def TextNumberSplit ( sString ):
    # der String muss mit Text beginnen !
    for iIndex, cLetter in enumerate ( sString, 0 ):
        if ( cLetter.isdigit () ):
            iNumber = sString[ iIndex : ]
            sText = sString[ : iIndex ]
            break
    if ( not sText ):
        sText = "No Success"
        iNumber = -1
    else:
        iNumber = int ( iNumber )
        
    return ( sText, iNumber )
# ********************************************************* Color Switcher *****************************************************************
def GetColor ( uIndex, bShow = False ):
    iNumEntries = 20
    sPalette = ""
    if ( uIndex == "none" ):
        return ( uIndex )
    if ( uIndex in [ "black", "white" ] ):
        return ( uIndex )
    if ( isinstance ( uIndex, ( str, np.str_ ) ) ):
        sPalette, iIndex = TextNumberSplit ( uIndex )
        CheckAssert ( bBool = ( iIndex != -1 ), sMsg = "Input Parsing failed!" )
    elif ( isinstance ( uIndex, ( tuple, list, np.ndarray ) ) ):
        return ( uIndex )
    else:
        iIndex = uIndex
  
    if ( bShow == False ):
        CheckAssert ( bBool =  ( ( iIndex < iNumEntries ) and ( iNumEntries <= 50 ) ), sMsg = "Inavalid Input Range for iIndex!" )
    if ( not sPalette ):
        sPalette = "tab20b"
    DictSwitcher = {      
        "red":      "Reds",
        "r":        "Reds",
        "green":    "Greens",
        "g":        "Greens",
        "blue":     "Blues",
        "b":        "Blues",
        "orange":   "Oranges",
        "o":        "Oranges",
        "copper":   "copper_r",
        "c":        "copper_r",
        "pink":     "RdPu",
        "p":        "RdPu",
        "silver":   "Greys",
        "s":        "Greys",
        "tab20b":   "tab20b",
        "tb":       "tab20b",
        "tab20c":   "tab20c",
        "tc":       "tab20c"
        }
    
    sColorMapName = DictSwitcher.get ( sPalette, lambda: "Invalid Input" )
    aColorMap = plt.get_cmap ( name = sColorMapName )
    aC = np.linspace ( 0.05, 0.95, iNumEntries )
    aColors = [ aColorMap ( j ) for j in aC ]

    if ( bShow == True ):
        for ik in range ( 0, iNumEntries ):
            print ( ik, aColors[ ik ] )
            plt.hlines ( aC[ ik ], 0, 1, color = aColors[ ik ] )
        iIndex = 1
    
    return ( aColors[ iIndex ] )

