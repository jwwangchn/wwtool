import numpy as np
from wwtool.datasets.spacenet import geoTools as gT
from osgeo import ogr
import os
from skimage.morphology import binary_dilation, disk
import shapely.geometry
from shapely import wkt
import cv2


def db_eval_boundary(foreground_mask, gt_mask, bound_th=3):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.
    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        bound_th: the distance threshold for calculating boundary F-score
    Returns:
        F (float): boundaries F-score
        P (float): boundaries precision
        R (float): boundaries recall
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))


    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask)
    gt_boundary = seg2bmap(gt_mask)

    fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary, disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall);

    return F


def seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
 """

    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) > 0.01), \
        'Can''t convert %dx%d seg to %dx%d bmap.' % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1;

    return bmap


def draw_poly(mask, poly):
    """
    Draw a polygon on the mask.
    """

    if not isinstance(poly, np.ndarray):
        poly = np.array(poly)
    cv2.fillPoly(mask, np.array([poly],dtype=np.int32), 255)

    return mask



def evaluate_instance(array_test_poly, array_truth_poly, img_size, bound_th=3):
    """
    Evaluate a single instance.
    """

    gt_mask = np.zeros((img_size,img_size),  dtype = np.uint8)
    pred_mask = np.zeros((img_size,img_size),  dtype = np.uint8)

    gt_mask = draw_poly(gt_mask, array_truth_poly)
    pred_mask = draw_poly(pred_mask, array_test_poly)

    F = db_eval_boundary(pred_mask, gt_mask, bound_th)

    return F


def boundfscore(test_poly, truth_polys, truth_index=[], img_size=2048, bound_th=3):

    fidlistArray = []
    bfs_list = []

    if truth_index:
        fidlist = gT.search_rtree(test_poly, truth_index)
        #print (fidlist)
        for fid in fidlist:
            #print (fid)
            if not test_poly.IsValid():
                #print (test_poly)
                test_poly = test_poly.Buffer(0.0)
                #print ("not valid")

            else:
                truth_poly = truth_polys[fid]
                fidlistArray.append(fid)
                
                if truth_poly.IsValid():

                    #print (img_size)
                    array_test_poly = gT.geoPolygon2nparray(test_poly)
                    array_truth_poly = gT.geoPolygon2nparray(truth_poly)

                    bfs = evaluate_instance(array_test_poly, array_truth_poly, img_size, bound_th)
                    bfs_list.append(bfs)

    else:

        for idx, truth_poly in enumerate(truth_polys):
            array_test_poly = gT.geoPolygon2nparray(test_poly)
            array_truth_poly = gT.geoPolygon2nparray(truth_poly)
            bfs = evaluate_instance(array_test_poly, array_truth_poly, img_size, bound_th)
            bfs_list.append(bfs)

            intersection_result = test_poly.Intersection(truth_poly)
            intersection_result.GetGeometryName()

              
    return bfs_list, fidlistArray


def iou(test_poly, truth_polys, truth_index=[]):
    fidlistArray = []
    iou_list = []
    if truth_index:
        fidlist = gT.search_rtree(test_poly, truth_index)

        for fid in fidlist:
            if not test_poly.IsValid():
                test_poly = test_poly.Buffer(0.0)

            intersection_result = test_poly.Intersection(truth_polys[fid])
            fidlistArray.append(fid)
             
            #print (test_poly)
            #print (truth_polys[fid])
            #print (intersection_result)            

            if not intersection_result:
                iou_list.append(0)

            elif intersection_result.GetGeometryName() == 'POLYGON' or \
                            intersection_result.GetGeometryName() == 'MULTIPOLYGON' or intersection_result.GetGeometryName() == 'GEOMETRYCOLLECTION':
                intersection_area = intersection_result.GetArea()
                union_area = test_poly.Union(truth_polys[fid]).GetArea()
                iou_list.append(intersection_area / union_area)

            else:
                iou_list.append(0)

    else:

        for idx, truth_poly in enumerate(truth_polys):
            intersection_result = test_poly.Intersection(truth_poly)
            intersection_result.GetGeometryName()

            if intersection_result.GetGeometryName() == 'POLYGON' or \
                            intersection_result.GetGeometryName() == 'MULTIPOLYGON':
                intersection_area = intersection_result.GetArea()
                union_area = test_poly.Union(truth_poly).GetArea()
                iou_list.append(intersection_area / union_area)

            else:

                iou_list.append(0)

    return iou_list, fidlistArray


def score(test_polys, truth_polys, threshold=0.5, truth_index=[],
          resultGeoJsonName = [],
          imageId = [], img_size=2048, bound_th=3):

    # Define internal functions

    # Find detections using threshold/argmax/IoU for test polygons
    true_pos_count = 0
    false_pos_count = 0
    truth_poly_count = len(truth_polys)

    if resultGeoJsonName:
        if not imageId:
            imageId = os.path.basename(os.path.splitext(resultGeoJsonName)[0])

        driver = ogr.GetDriverByName('geojson')
        if os.path.exists(resultGeoJsonName):
            driver.DeleteDataSource(resultGeoJsonName)
        datasource = driver.CreateDataSource(resultGeoJsonName)
        layer = datasource.CreateLayer('buildings', geom_type=ogr.wkbPolygon)
        field_name = ogr.FieldDefn("ImageId", ogr.OFTString)
        field_name.SetWidth(75)
        layer.CreateField(field_name)
        layer.CreateField(ogr.FieldDefn("BuildingId", ogr.OFTInteger))
        layer.CreateField(ogr.FieldDefn("IOUScore", ogr.OFTReal))


    iou_imgtotal = 0 ###
    bfs_imgtotal = 0

    print (imageId)

    for test_poly in test_polys:

        #print (test_poly)
        #print (truth_polys)
        
        if truth_polys:
            
            bfs_list, bfs_fidlist = boundfscore(test_poly, truth_polys, truth_index, img_size, bound_th)
            iou_list, fidlist = iou(test_poly, truth_polys, truth_index)

            ### calculate the total iou of all buildings in an image
            if not iou_list:
                maxiou = 0
            else:
                maxiou = np.max(iou_list)
            iou_imgtotal = iou_imgtotal + maxiou ###

            ### calculate the total bfs of all buildings in an image
            if not bfs_list:
                maxbfs = 0
            else:
                maxbfs = np.max(bfs_list) 
            bfs_imgtotal = bfs_imgtotal + maxbfs ### 

            ### decide the type of test_poly TP/FP/FN based on its maxiou
            if maxiou >= threshold:
                true_pos_count += 1
                truth_index.delete(fidlist[np.argmax(iou_list)], truth_polys[fidlist[np.argmax(iou_list)]].GetEnvelope())
                #del truth_polys[fidlist[np.argmax(iou_list)]]
                if resultGeoJsonName:
                    feature = ogr.Feature(layer.GetLayerDefn())
                    feature.SetField('ImageId', imageId)
                    feature.SetField('BuildingId', fidlist[np.argmax(iou_list)])
                    feature.SetField('IOUScore', maxiou)
                    feature.SetGeometry(test_poly)
                
                #print ('true positive: ' + str(test_poly))

            else:
                false_pos_count += 1

                if resultGeoJsonName:
                    feature = ogr.Feature(layer.GetLayerDefn())
                    feature.SetField('ImageId', imageId)
                    feature.SetField('BuildingId', -1)
                    maxiou = float(maxiou) ###
                    feature.SetField('IOUScore', maxiou)
                    feature.SetGeometry(test_poly)

                #print(('false positive: ' + str(test_poly)))



        else:
            false_pos_count += 1

            if resultGeoJsonName:
                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetField('ImageId', imageId)
                feature.SetField('BuildingId', 0)
                feature.SetField('IOUScore', 0)
                feature.SetGeometry(test_poly)

        if resultGeoJsonName:
            layer.CreateFeature(feature)
            feature.Destroy()

    if resultGeoJsonName:
        datasource.Destroy()


    false_neg_count = truth_poly_count - true_pos_count


    return true_pos_count, false_pos_count, false_neg_count, iou_imgtotal, bfs_imgtotal


def evalfunction(xxx_todo_changeme,
                 resultGeoJsonName=[],
                 threshold = 0.5,
                 img_size=2048,
                 bound_th=3):


    (image_id, test_polys, truth_polys, truth_index) = xxx_todo_changeme
    if len(truth_polys)==0:
        true_pos_count = 0
        false_pos_count = len(test_polys)
        false_neg_count = 0
        iou_imgtotal = 0
        bfs_imgtotal = 0
    else:
        true_pos_count, false_pos_count, false_neg_count, iou_imgtotal, bfs_imgtotal = score(test_polys, truth_polys.tolist(),
                                                                 truth_index=truth_index,
                                                                 resultGeoJsonName=resultGeoJsonName,
                                                                 imageId=image_id,
                                                                 threshold=threshold,
                                                                 img_size=img_size,
                                                                 bound_th=bound_th
                                                                 )


    if (true_pos_count > 0):

        precision = float(true_pos_count) / (float(true_pos_count) + float(false_pos_count))
        recall = float(true_pos_count) / (float(true_pos_count) + float(false_neg_count))
        F1score = 2.0 * precision * recall / (precision + recall)
    else:
        F1score = 0


    #return ((F1score, true_pos_count, false_pos_count, false_neg_count), image_id)
    return ((F1score, true_pos_count, false_pos_count, false_neg_count, iou_imgtotal, bfs_imgtotal), image_id) ### lwj changed


def  create_eval_function_input(xxx_todo_changeme1):

    (image_ids, (prop_polysIdList, prop_polysPoly), (sol_polysIdsList, sol_polysPoly)) = xxx_todo_changeme1
    evalFunctionInput = []


    for image_id in image_ids:
        test_polys = prop_polysPoly[np.argwhere(prop_polysIdList == image_id).flatten()]
        truth_polys = sol_polysPoly[np.argwhere(sol_polysIdsList == image_id).flatten()]
        truth_index = gT.create_rtree_from_poly(truth_polys)
        evalFunctionInput.append([image_id, test_polys, truth_polys, truth_index])

    return evalFunctionInput


