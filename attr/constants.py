"""
Filename: constants.py
Author(s): Kayvon Khosrowpour
Date Created: 11/2/18

Description:
Contains constants for attribute extraction.
"""

class FrameColumns:
	info = ['filename', 'path']
	median_gray = ['medianGray']
	median_hsv = ['medianHue', 'medianSat', 'medianVal']
	entropy = ['entropy']
	kmeans = ['TODO']

class TruthColumns:
	info = ['filename']
	dropna_columns = ['style']
	keep_styles = [
		'Medieval',
		'Renaissance',
		'Baroque',
		'Romanticism',
		'Realism',
		'Impressionism',
		'Post-Impressionism',
		'Art Nouveau (Modern)',
		'Expressionism',
		'Abstract Expressionism',
		'Contemporary'
	]
	drop_columns = ['artist', 'title', 'genre', 'date']