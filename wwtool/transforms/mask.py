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

    # delete invalid polygon
    for idx, polygon in enumerate(polygons):
        if not polygon.is_valid:
            polygons.pop(idx)

    polygons = geopandas.GeoDataFrame({'geometry': polygons, 'polygon_df':range(len(polygons))})

    clipped_polygons = geopandas.clip(polygons, image_boundary_polygon).to_dict()
    clipped_polygons = list(clipped_polygons['geometry'].values())

    results = []
    for clipped_polygon in clipped_polygons:
        if type(clipped_polygon) == MultiPolygon:
            for single_polygon in clipped_polygon:
                results.append(single_polygon)
        elif type(clipped_polygon) == Polygon:
            results.append(clipped_polygon)
        else:
            continue

    return results

def clip_mask(masks, image_size=(1024, 1024)):
    polygons = [mask2polygon(mask) for mask in masks]
    clipped_polygons = clip_polygon(polygons, image_size)
    masks = [polygon2mask(clipped_polygon) for clipped_polygon in clipped_polygons]

    return masks