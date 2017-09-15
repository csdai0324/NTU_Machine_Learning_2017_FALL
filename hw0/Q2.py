import sys
from PIL import Image

def half_pixel_value(img):

	width, height = img.size
	pixels = list(img.getdata())
	new_pixels = []
	for pixel_value in pixels:
		new_pixels.append(tuple([int(value/2) for value in list(pixel_value)]))
	new_img = Image.new('RGB', (width, height))
	new_img.putdata(new_pixels)
	new_img.save('Q2.jpg')

if __name__ == '__main__':

	new_img = half_pixel_value(Image.open(sys.argv[1]))
