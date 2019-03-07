from operation import *

def delineate(cfg, dem_ds, outlet_ds):
    """
    Delineate watershed boundary based on the outlet location
    """
    # Flow Direction
    no_sinks_ds = RemoveSinksOp(cfg, input_ds=dem_ds).save()
    flow_dir_ds = FlowDirectionOp(cfg, input_ds=no_sinks_ds).save()
    flow_acc_ds = FlowAccumulationOp(cfg, input_ds=flow_dir_ds).save()
    
    stream_def_ds = StreamDefinitionByThresholdOp(
            cfg, flow_accum_ds=flow_acc_ds
    ).save()

    snap_outlet_ds = MoveOutletsToStreamOp(
            cfg,
            flowdir_ds = flow_dir_ds,
            stream_ds = stream_def_ds,
            outlet_ds = outlet_ds
    ).save()
    
    label_outlet_ds = LabelGagesOp(cfg, outlet_ds=snap_outlet_ds).save()
    boundary_raster_ds = GageWatershedOp(
            cfg,
            flowdir_ds = flow_dir_ds,
            outlet_ds = label_outlet_ds
    ).save()
    
    boundary_ds = RasterToShapefileOp(
            cfg, input_ds=boundary_raster_ds).saveas('shp', working_ext='shp')
    return boundary_ds

