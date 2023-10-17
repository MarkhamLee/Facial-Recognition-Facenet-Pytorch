# a a couple of helper functions for the "front end" notebook for opening
# image files and displaying themm with a green or red tinting depending
# on whether or not the photos match. This code could easily be in the
# notebook, but I think it's cleaner to place it in a separate python
# script

from PIL import Image
import matplotlib.pyplot as plt


class TestFunctions:

    # this method opens the images and returns the image data
    def open_images(self, first, second):

        with Image.open(first) as photo1:
            photo1.load()

        with Image.open(second) as photo2:
            photo2.load()

        return photo1, photo2

    # method to display two photos side by side
    def show_images(self, imagea, imageb, title="Evaluated Photo"):

        # create canvas that photos will be plotted on
        figure = plt.figure(figsize=(8, 8))

        figure.add_subplot(1, 2, 1)
        plt.title("Reference Photo")
        plt.imshow(imagea)

        figure.add_subplot(1, 2, 2)
        plt.title(title)
        plt.imshow(imageb)

    # this method puts a red or green tint on the evaluated photo depending
    # on the match status expected input is two images that have been opened
    # with the Python Pillow (PIL) library result is a 1 for matched photos
    # 0 for photos that didn't match
    def display_results(self, imagea, imageb, result):

        # create required tint: red for no match, green for match
        if result == 0:
            tint = Image.new("RGB", (imageb.size), color=(255, 155, 15))
            title = 'No Match!'

        else:
            tint = Image.new("RGB", (imageb.size), color=(55, 255, 11))
            title = 'Photos Match!'

        # with the tint determined, blend the photo and the tint together
        comparedPhoto = Image.blend(imageb, tint, 0.5)

        # push the reference image and the sample photo to the method that
        # will display the images
        self.show_images(imagea, comparedPhoto, title)
