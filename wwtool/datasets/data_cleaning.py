import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import geopandas
from shapely.geometry import Polygon, MultiPolygon

import wwtool


def cleaning_polygon_by_polygon(origin_polygons, ignore_polygons, show=False):
    origin_polygons = wwtool.clean_polygon(origin_polygons)
    ignore_polygons = wwtool.clean_polygon(ignore_polygons)

    foot_polygons = geopandas.GeoSeries(origin_polygons)
    ignore_polygons = geopandas.GeoSeries(ignore_polygons)

    foot_df = geopandas.GeoDataFrame({'geometry': foot_polygons, 'foot_df':range(len(foot_polygons))})
    ignore_df = geopandas.GeoDataFrame({'geometry': ignore_polygons, 'ignore_df':range(len(ignore_polygons))})

    if show:
        fig, ax = plt.subplots(1, 2)

        ignore_df.plot(ax=ax[0], color='red')
        foot_df.plot(ax=ax[0], facecolor='none', edgecolor='k')
        ax[0].set_title('Before ignoring')

    # find the intersection between origin polygon and ignore polygon, get the index
    res_intersection = geopandas.overlay(foot_df, ignore_df, how='intersection')
    inter_dict = res_intersection.to_dict()
    ignore_indexes = list(set(inter_dict['foot_df'].values()))
    ignore_indexes.sort()

    converted_polygon = origin_polygons[:]
    # delete ignore index
    for ignore_index in ignore_indexes[::-1]:
        converted_polygon.pop(ignore_index)

    if show:
        foot_polygons = geopandas.GeoSeries(converted_polygon)
        foot_df = geopandas.GeoDataFrame({'geometry': foot_polygons, 'foot_df':range(len(foot_polygons))})
        ignore_df.plot(ax=ax[1], color='red')
        foot_df.plot(ax=ax[1], facecolor='none', edgecolor='k')
        ax[1].set_title('After ignoring')
        # plt.axis('off')

        plt.savefig('./a.png', bbox_inches='tight', dpi=600, pad_inches=0.5)

        plt.show()

    return converted_polygon, ignore_indexes