"""
File: kmeans.py
Author(s): Kayvon Khosrowpour
Date created: 10/23/18

Description:
Provides methods to collect kmeans data on an image. Useful for determining
dominant colors.
"""

import cv2
import numpy as np

def kmeans(img):
    """
    Applies kmeans clustering to a single img and returns the kmeans img and
    a list of the data gathered by the kmeans calculation.

    Args:
      img: np.array representing the img, as read from imread()

    Output:
      clusters: list of tuples of the 5 clusters found by kmeans. Each entry
         in this list contains the (num_pixels_with_this_color, color)
         sorted by dominance.
            For example, clusters[0] is a tuple of the format:
                (num_pixels_with_this_color, color)
            Color is the bgr np.array representing the color of the cluster.

    Notes:
      clusters[0] is the most dominant cluster
        ...
      clusters[k-1] is the the LEAST dominant cluster
    """
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    # BEST CRITERIA FROM KMEANS PERMUTATIONS
    max_iterations = 3
    epsilon = 1.0
    attempts = 2
    flags = cv2.KMEANS_PP_CENTERS
    K = 5

    criteria = (cv2.TERM_CRITERIA_MAX_ITER, max_iterations, epsilon)
    ret,label,centers = cv2.kmeans(Z,K,None,criteria,attempts,flags)

    # Now convert back into uint8, and make original image
    palette = np.uint8(centers)
    quantized = palette[label.flatten()]
    kmeaned_img = quantized.reshape((img.shape))

    # initialize clustered imgs to return
    clusters = []

    # get data for each cluster
    for center in centers:
        # convert the float value of center to a BGR color value
        color = center.astype(np.uint8)

        # find all pixel values in the kmeans image that correspond to this color
        pixels_with_this_color = np.where(kmeaned_img==color)

        # num_pixels_with_this_color gives an idea of the relative presence
        # of this color compared to other colors
        num_pixels_with_this_color = pixels_with_this_color[0].size / 3

        # add to list with number of pixels in img with this color
        clusters.append((num_pixels_with_this_color, color))

    # sort the clusters by dominance
    clusters = sorted(clusters, reverse=True)

    return clusters

def kmeans_stats(clusters):
    """
    Given a list of tuples of clusters (obtained from kmeans()),
    provides relevant statistics about the clustering.

    Arg:
      clusters: list of tuples of the 5 clusters found by kmeans. Each entry
         in this list contains the (num_pixels_with_this_color, color)
         sorted by dominance.
            For example, clusters[0] is a tuple of the format:
                (num_pixels_with_this_color, color)
            Color is the bgr np.array representing the color of the cluster.

    Output:
      dist_btwn_cluster2_and_black: distance between the vector colors provided
        by clusters
      ratio_top2_clusters: ratio of the number of pixels in the most dominant
        cluster to the second most dominant color
      ratio_last2_clusters: ratio of the number of pixels in the second least
        dominant cluster to the least dominant color
      dominant_bgr_color2_num_px: number of pixels in the second most
        dominant cluster
      dominant_bgr_color1_num_px: number of pixels in the first most
        dominant cluster
      dominant_bgr_color4_num_px: number of pixels in the fourth most
        dominant cluster
      ratio_top2_to_last2_clusters: ratio of the number of pixels in the top two
        to the last 2 most dominant clusters
      dist_btwn_top_2_vectors: the euclidean distance between the top two most
        dominant cluster colors
    """

    # Note that not all clusters are needed for model, so some
    # clusters are ignored.
    cluster1_color = np.array(clusters[0][1], dtype=np.int32)
    cluster1_num_px = clusters[0][0]

    cluster2_color = np.array(clusters[1][1], dtype=np.int32)
    cluster2_num_px = clusters[1][0]

    cluster4_color = np.array(clusters[3][1], dtype=np.int32)
    cluster4_num_px = clusters[3][0]

    cluster5_color = np.array(clusters[4][1], dtype=np.int32)
    cluster5_num_px = clusters[4][0]

    # distance between cluster 2's color and black in the BGR color space
    dist_btwn_cluster2_and_black = np.linalg.norm(cluster2_color - black)

    # ratios of the number of pixels in the clusters
    ratio_top2_clusters = cluster1_num_px / cluster2_num_px
    ratio_last2_clusters = cluster4_num_px / cluster5_num_px

    # get number of pixels for each
    dominant_bgr_color2_num_px = cluster2_num_px
    dominant_bgr_color1_num_px = cluster1_num_px
    dominant_bgr_color4_num_px = cluster4_num_px

    # get the ratio of the num of pixels in top 2 dominant clusters to the last 2
    ratio_top2_to_last2_clusters = (cluster1_num_px + cluster2_num_px) / (cluster4_num_px + cluster5_num_px)

    # get distance between the top two vectors
    dist_btwn_top_2_vectors = np.linalg.norm(cluster1_color - cluster2_color)

    return dist_btwn_cluster2_and_black,\
           ratio_top2_clusters,\
           ratio_last2_clusters,\
           dominant_bgr_color2_num_px,\
           dominant_bgr_color1_num_px,\
           dominant_bgr_color4_num_px,\
           ratio_top2_to_last2_clusters,\
           dist_btwn_top_2_vectors

