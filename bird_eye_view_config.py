class BaseConfig(object):
    TimeStamp = "MTS.Package.TimeStamp"
    eObjMaintenanceState = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].General.eMaintenanceState"
    uiLifeTime = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].General.fLifeTime"
    DistX = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].Kinematic.fDistX"
    DistY = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].Kinematic.fDistY"
    VrelX = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].Kinematic.fVrelX"
    VrelY = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].Kinematic.fVrelY"
    ArelX = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].Kinematic.fArelX"
    ucDynamicProperty = "SIM VFB ALL.FPS.FPS_ARSObjectListMeas.aObject[{}].MotionAttributes.eDynamicProperty"
    ucDynamicSubProperty = "SIM VFB ALL.FPS.FPS_ARSObjectListMeas.aObject[{}].MotionAttributes.eDynamicSubProperty"
    StoppedConfidence = "SIM VFB ALL.FPS.FPS_ARSObjectListMeas.aObject[{}].MotionAttributes.uiStoppedConfidence"
    Orient = "SIM VFB ALL.FPS.FPS_ARSObjectListMeas.aObject[{}].Geometry.fOrientation"
    Object_Length = "SIM VFB ALL.FPS.FPS_ARSObjectListMeas.aObject[{}].Geometry.fLength"
    Object_Width = "SIM VFB ALL.FPS.FPS_ARSObjectListMeas.aObject[{}].Geometry.fWidth"
    uiID = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].General.uiID"
    EBAQ = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].Qualifiers.uiEbaObjQuality"
    ObstDtct = "SIM VFB ALL.FPS.FPS_CustomObjectListMeas.CustObjects[{}].LRR_Obj_ObstclDtct"
    FuncQual = "SIM VFB ALL.FPS.FPS_CustomObjectListMeas.CustObjects[{}].LRR_Obj_CollisionFunc_Qual"
    ObjPreSelect = "SIM VFB ALL.AlgoSenCycle.FCTCustomOutputData.CustObjData[{}].LRR_ObjData.LRR_Obj_ObjPreSelect"
    Overlap = "SIM VFB ALL.AlgoSenCycle.FCTCustomOutputData.CustObjData[{}].LRR_ObjData.LRR_Obj_Collision_Ovrlp"
    Staight = "SIM VFB ALL.AlgoSenCycle.FCTCustomOutputData.CustObjData[{}].LRR_ObjData.LRR_Obj_Straight"
    DistToRef = "SIM VFB ALL.AlgoSenCycle.FCTPublicObjData.ObjList[{}].Legacy.fDistToRef"
    ObservedClass = "SIM VFB ALL.AlgoSenCycle.FCTCustomOutputData.CustObjData[{}].LRR_ObjData.LRR_Obj_2_Observed_Class"
    ObservedHist = "SIM VFB ALL.FPS.FPS_CustomObjectListMeas.CustObjects[{}].LRR_Obj_ObsrvHistory"
    PoE = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].Qualifiers.uiProbabilityOfExistence"
    fRCS = "SIM VFB ALL.FPS.FPS_ARSObjectListMeas.aObject[{}].SensorSpecific.fRCS"
    PedProb = "SIM VFB ALL.FPS.FPS_CustomObjectListMeas.CustObjects[{}].LRR_Obj_Pedestrian_Qual"
    BicProb = "SIM VFB ALL.FPS.FPS_CustomObjectListMeas.CustObjects[{}].LRR_Obj_Bicycle_Qual"
    Classification = "SIM VFB ALL.FPS.FPS_ARSObjectListMeas.aObject[{}].Attributes.eClassification"
    eLane = "SIM VFB ALL.AlgoSenCycle.FCTPublicObjData.ObjList[{}].LaneInformation.eAssociatedLane"
    objLane = "SIM VFB ALL.AlgoSenCycle.FCTCustomOutputData.CustObjData[{}].LRR_ObjData.LRR_Obj_Lane"
    VELOCITY = "SIM VFB ALL.AlgoVehCycle.VehDyn.Longitudinal.MotVar.Velocity"
    ACCELERATION = "SIM VFB ALL.AlgoVehCycle.VehDyn.Longitudinal.MotVar.Accel"
    YAW_RATE = "SIM VFB ALL.AlgoVehCycle.VehDyn.Lateral.YawRate.YawRate"
    SLIP_ANGLE = "SIM VFB ALL.AlgoVehCycle.VehDyn.Lateral.SlipAngle.SideSlipAngle"
    VDY_CURVE_SIGNAL = "SIM VFB ALL.DAP.DAP_VehDynTgtSyncNearMeas.Lateral.Curve.Curve"

class CustomBirdEyeSettings(BaseConfig):
    """ For the Innovation structure the object list interfaces changed
    """
    """
    Object signals without simulator name (SIM VFB, SIM VFB EM, e.g.) and
    without Cycle name (DataProcCycle, EM, e.g.) otherwise they have to be adapted with for every device.
    """
    MAINTENANCE_STATE = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].General.eMaintenanceState"
    LIFECYCLES = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].General.uiLifeCycles"
    UIID = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].General.uiID"

    DIST_X = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].Kinematic.fDistX"
    DIST_Y = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].Kinematic.fDistY"

    # Shapepoints -> Default
    SHAPE_POINT_X = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].Geometry.aShapePointCoordinates[{}].fPosX"
    SHAPE_POINT_Y = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].Geometry.aShapePointCoordinates[{}].fPosY"

    # L - Shape Points
    # Reference_Point
    REF_DIST_X = "SIM VFB ALL.FPS.FPS_CustomObjectListMeas.CustObjects[{}].LShapePoints.sRefPoint.fXDist"
    REF_DIST_Y = "SIM VFB ALL.FPS.FPS_CustomObjectListMeas.CustObjects[{}].LShapePoints.sRefPoint.fYDist"

    # L - Left
    Left_SHAPE_POINT_X = "SIM VFB ALL.FPS.FPS_CustomObjectListMeas.CustObjects[{}].LShapePoints.sDeltaLeft.fXDist"
    Left_SHAPE_POINT_Y = "SIM VFB ALL.FPS.FPS_CustomObjectListMeas.CustObjects[{}].LShapePoints.sDeltaLeft.fYDist"

    # L - Middle
    Middle_SHAPE_POINT_X = "SIM VFB ALL.FPS.FPS_CustomObjectListMeas.CustObjects[{}].LShapePoints.sDeltaMiddle.fXDist"
    Middle_SHAPE_POINT_Y = "SIM VFB ALL.FPS.FPS_CustomObjectListMeas.CustObjects[{}].LShapePoints.sDeltaMiddle.fYDist"

    # L - Right
    Right_SHAPE_POINT_X = "SIM VFB ALL.FPS.FPS_CustomObjectListMeas.CustObjects[{}].LShapePoints.sDeltaRight.fXDist"
    Right_SHAPE_POINT_Y = "SIM VFB ALL.FPS.FPS_CustomObjectListMeas.CustObjects[{}].LShapePoints.sDeltaRight.fYDist"

    # ARS4xx width/length/orientation -> fallback
    WIDTH = "SIM VFB ALL.FPS.FPS_ARSObjectListMeas.aObject[{}].Geometry.fWidth"
    LENGTH = "SIM VFB ALL.FPS.FPS_ARSObjectListMeas.aObject[{}].Geometry.fLength"
    ORIENTATION = "SIM VFB ALL.FPS.FPS_ARSObjectListMeas.aObject[{}].Geometry.fOrientation"

    # Else 1 x 1 m objects

    # radaronly...
    EM_DYN_PROP = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].Attributes.eDynamicProperty"
    ARS_DYN_PROP = "SIM VFB ALL.FPS.FPS_ARSObjectListMeas.aObject[{}].MotionAttributes.eDynamicProperty"

    # Relative velocity
    V_REL_X = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].Kinematic.fVrelX"
    V_REL_Y = "SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{}].Kinematic.fVrelY"


def get_unique_signals():
    # Extract signals from BaseConfig and CustomBirdEyeSettings
    base_signals = {k: v for k, v in BaseConfig.__dict__.items() if not k.startswith("__")}
    custom_signals = {k: v for k, v in CustomBirdEyeSettings.__dict__.items() if not k.startswith("__")}

    # Merge dictionaries, preferring custom signals over base signals for any duplicate keys
    unique_signals = {**base_signals, **custom_signals}

    return unique_signals