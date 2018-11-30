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

# constants
BLACK = np.array([0, 0, 0], dtype=np.int32)

def kmeans(img, K, max_iterations=3, epsilon=1.0, attempts=2, flags=cv2.KMEANS_PP_CENTERS):
    """
    Applies kmeans clustering to a single img and returns the kmeans img and
    a list of the data gathered by the kmeans calculation.

    Args:
        img: np.array representing the img, as read from imread()
        K: no shit

    Output:
        clusters: list of tuples of the 5 clusters found by kmeans. Each entry
             in this list contains the (log_of_num_px, color)
             sorted by dominance.
                    For example, clusters[0] is a tuple of the format:
                            (log_of_num_px, color)
                    Color is the bgr np.array representing the color of the cluster.
        kmeaned_img: the clustered image.

    Notes:
        clusters[0] is the most dominant cluster
            ...
        clusters[k-1] is the the LEAST dominant cluster
    """
    assert (K >= 2)

    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria and run kmeans
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, max_iterations, epsilon)
    ret, label, centers = cv2.kmeans(Z,K,None,criteria,attempts,flags)

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
            num_pixels_with_this_color = round(pixels_with_this_color[0].size / 3) # b/c of bgr
            log_of_num_px = np.log(num_pixels_with_this_color)

            # add to list with number of pixels in img with this color
            clusters.append((log_of_num_px, color))

    # sort the clusters by dominance
    clusters = sorted(clusters, reverse=True, key=lambda x: clusters[0])

    return clusters, kmeaned_img

def kmeans_stats(clusters, dist_colors=np.array([BLACK])):
    """
    Given a list of tuples of clusters (obtained from kmeans()),
    provides relevant statistics about the clustering.

    Argument(s):
        clusters: list of tuples of the k clusters found by kmeans.
            For example, clusters[0] is a tuple of the format:
                    (log_of_num_px, color)
            Color is the bgr np.array representing the color of the cluster.
        dist_colors: the numpy array of BGRcolors of which to calculate the distances of.
            For example suppose len(clusters) = 3. If `dist_colors` = [[0, 0, 0]], 
            then the return value `color_distances` is a numpy matrix, where
            color_distances[0] is the list of euclidean distances between black (i.e. [0, 0, 0])
            and the 0th color from `cluster`.

    Output (indexed identically to clusters argument):
        num_px: the numpy array of number of pixels in each cluster
        colors: the vector colors for the clusters
        dist_top_2_vectors: the euclidean distance between the top two most
            dominant cluster colors
        ratio_top2_clusters: ratio of the number of pixels in the top two clusters
        ratio_last2_clusters: ratio of the number of pixels in the bottom two clusters
        color_distances: a numpy matrix, where each row is the list of distances between
            each cluster color and one of the provided colors.
    """

    assert (len(clusters) >= 2)

    # get num_px and colors directly from the clusters
    num_px = np.array([cluster[0] for i,cluster in enumerate(clusters)])
    colors = np.array([np.array(cluster[1], dtype=np.int32) for i,cluster in enumerate(clusters)])

    # calculate the distance between the top 2 vectors
    dist_top_2_vectors = np.linalg.norm(colors[0] - colors[1])

    # calculate the ratio between number of pixels clustered in the top 2 vectors
    ratio_top2_clusters = num_px[0] / num_px[1]

    # calculate the ratio between number of pixels clustered in the bottom 2 vectors
    ratio_last2_clusters = num_px[-2] / num_px[-1]

    # calculate distances to black
    color_distances = []
    for color in colors: # for each clustered color
        distances = []
        for d_color in dist_colors: # compute distance between the provided distance colors
            distances.append(np.linalg.norm(color - d_color))
        color_distances.append(np.array(distances))
    color_distances = np.array(color_distances, dtype=np.int32)

    return num_px,\
            colors,\
            dist_top_2_vectors,\
            ratio_top2_clusters,\
            ratio_last2_clusters,\
            color_distances

if __name__ == '__main__':
    img = cv2.imread('/Users/kayvon/code/divp/proj/train/train_1/1.jpg', cv2.IMREAD_COLOR)
    clusters, kmeaned_img = kmeans(img, 5)

    dist_colors = np.array([BLACK, np.array([255, 255, 255])])

    num_px, colors, dt2v, rt2c, rl2c, c_dist = kmeans_stats(clusters, dist_colors=dist_colors)

    for (ct, color) in zip(num_px, colors):
        print('Color', color, ' \t\t has', ct, 'pixels')

    print('Distance between top 2 vectors:', dt2v)
    print('Ratio between top 2 clusters:', rt2c)
    print('Ratio between bottom 2 clusters:', rl2c)
    print('Distances between clusters and dist_colors:')
    print(c_dist)

    the_input = input('Show? ')
    if the_input.lower() == 't':
        cv2.imshow('clustered', kmeaned_img)
        cv2.waitKey(5000)

