"""
Filename: get.py
Author(s): Kayvon Khosrowpour

Description:
Copies images from a source location that are of interest
to the art classifier. We are only interested in the styles
defined in `keep_styles` in this script.
"""

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