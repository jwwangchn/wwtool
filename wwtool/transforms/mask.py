import shapely
from shapely.geometry import Polygon, MultiPolygon
import geopandas
import geojson


def mask2polygon(mask):
    """convert mask to polygon

    Arguments:
        mask {list} -- contains coordinates of mask boundary ([x1, y1, x2, y2, ...])
    """
    mask_x = mask[0::2]
    mask_y = mask[1::2]
    mask_coord = [(x, y) for x, y in zip(mask_x, mask_y)]

    polygon = Polygon(mask_coord)
    if not polygon.is_valid:
        # print("invalid polygon: ", polygon)
        polygon = polygon.buffer(0)
        if type(polygon) == Polygon:
            polygon = Polygon(polygon.exterior.coords)
        else:
            polygon = Polygon(polygon[0].exterior.coords)

        # print("fixed polygon: ", polygon)
    
        if polygon.wkt == 'GEOMETRYCOLLECTION EMPTY':
            return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    return polygon

def polygon2mask(polygon):
    """convet polygon to mask

    Arguments:
        polygon {Polygon} -- input polygon (single polygon)

    Returns:
        list -- converted mask ([x1, y1, x2, y2, ...])
    """
    geo = geojson.Feature(geometry=polygon, properties={})
    if geo.geometry == None:
        return []
    coordinate = geo.geometry["coordinates"][0]     # drop the polygon of hole
    mask = []
    for idx, point in enumerate(coordinate):
        if idx == len(coordinate) - 1:
            break
        x, y = point
        mask.append(int(x))
        mask.append(int(y))
    return mask

def clip_polygon(polygons, image_size=(1024, 1024)):
    h, w = image_size
    image_boundary_polygon = Polygon([(0, 0), (w-1, 0), (w-1, h-1), (0, h-1), (0, 0)])

    polygons = clean_polygon(polygons)

    polygons = geopandas.GeoDataFrame({'geometry': polygons, 'polygon_df':range(len(polygons))})

    clipped_polygons = geopandas.clip(polygons, image_boundary_polygon).to_dict()
    clipped_polygons = list(clipped_polygons['geometry'].values())

    clipped_polygons = clean_polygon(clipped_polygons)

    return clipped_polygons

def clip_mask(masks, image_size=(1024, 1024)):
    polygons = [mask2polygon(mask) for mask in masks]
    clipped_polygons = clip_polygon(polygons, image_size)
    masks = [polygon2mask(clipped_polygon) for clipped_polygon in clipped_polygons]

    return masks


def clean_polygon(polygons):
    """convert polygon to valid polygon

    Arguments:
        polygons {list} -- list of polygon

    Returns:
        list -- cleaned polygons
    """
    polygons_ = []
    for polygon in polygons:
        if not polygon.is_valid:
            continue
        if type(polygon) == MultiPolygon:
            for single_polygon in polygon:
                if len(list(single_polygon.exterior.coords)) < 3:
                    continue
                polygons_.append(single_polygon)
        elif type(polygon) == Polygon:
            if len(list(polygon.exterior.coords)) < 3:
                continue
            polygons_.append(polygon)
        else:
            continue

    return polygons_