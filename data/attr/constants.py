"""
Filename: constants.py
Author(s): Kayvon Khosrowpour
Date Created: 11/2/18

Description:
Contains constants for attribute extraction.
"""
import numpy as np

class FrameColumns:
	info = ['filename', 'path']
	median_gray = ['median_gray']
	median_hsv = ['median_hue', 'median_sat', 'median_val']
	entropy = ['entropy']
	kmeans = ['dist_top_2_vectors', 'log_ratio_top2_clusters', 'log_ratio_last2_clusters']
	average_edges = ['canny1', 'canny2', 'canny3', 'canny4']
	x_y_gradient = ['x_gradient', 'y_gradient']
	freq_bands = ['freqBand1E','freqBand1M', 'freqBand1S', 'freqBand2E','freqBand2M','freqBand2S',
	             'freqBand3E', 'freqBand3M', 'freqBand3S','freqBand4E','freqBand4M', 'freqBand4S',
				 'freqBand5E', 'freqBand5M', 'freqBand5S']
	stats = ['standard_deviation']
	freq_stats = ['freq_standard_deviation']
	num_px = None	# set in attr_extr
	color = None	# set in attr_extr
	dist_cluster_colors = None	# set in attr_extr

# colors to calculate the distance betwen clusters
BLACK = np.array([0,0,0], dtype=np.int32)
DISTANCE_COLORS = np.array([BLACK], dtype=np.int32)

class TruthColumns:
	info = ['filename']
	dropna_columns = ['style']
	keep_styles = [
		'Medieval',
		'Renaissance',
		'Baroque',
		'Realism',
		'Impressionism',
		'Post-Impressionism',
		'Art Nouveau (Modern)',
		'Expressionism',
		'Abstract Expressionism',
		'Contemporary'
	]
