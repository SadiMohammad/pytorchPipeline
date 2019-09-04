import os
import logging

import numpy as np
from imageio import imread
import matplotlib
from matplotlib import pyplot as plt

import morphsnakes as ms

class morphSnake:
    def __init__(self, maskPred, center, dia):
        self.maskPred = maskPred
        self.center = center
        self.dia = dia

    def visual_callback_2d(background, fig=None):
        """
        Returns a callback than can be passed as the argument `iter_callback`
        of `morphological_geodesic_active_contour` and
        `morphological_chan_vese` for visualizing the evolution
        of the levelsets. Only works for 2D images.
        
        Parameters
        ----------
        background : (M, N) array
            Image to be plotted as the background of the visual evolution.
        fig : matplotlib.figure.Figure
            Figure where results will be drawn. If not given, a new figure
            will be created.
        
        Returns
        -------
        callback : Python function
            A function that receives a levelset and updates the current plot
            accordingly. This can be passed as the `iter_callback` argument of
            `morphological_geodesic_active_contour` and
            `morphological_chan_vese`.
        
        """
        
        # Prepare the visual environment.
        if fig is None:
            fig = plt.figure()
        fig.clf()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(background, cmap=plt.cm.gray)

        ax2 = fig.add_subplot(1, 2, 2)
        ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
        plt.pause(0.001)

        def callback(levelset):
            
            if ax1.collections:
                del ax1.collections[0]
            ax1.contour(levelset, [0.5], colors='r')
            ax_u.set_data(levelset)
            fig.canvas.draw()
            plt.pause(0.001)

        return callback

    def rgb2gray(img):
        """Convert a RGB image to gray scale."""
        return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    def example_lakes():
        logging.info('Running: example_lakes (MorphACWE)...')
        
        # Load the image.
        img = self.maskPred
        
        # MorphACWE does not need g(I)
        
        # Initialization of the level-set.
        init_ls = ms.circle_level_set(img.shape, (self.center[0], self.center[1]), (self.dia)/4)
        
        # Callback for visual plotting
        callback = visual_callback_2d(img)

        # Morphological Chan-Vese (or ACWE)
        ms.morphological_chan_vese(img, iterations=200,
                                   init_level_set=init_ls,
                                   smoothing=3, lambda1=1, lambda2=1,
                                   iter_callback=callback)


if __name__ == '__USexp__':
	logging.basicConfig(level=logging.DEBUG)
	example_lakes()

	# Uncomment the following line to see a 3D example
	# This is skipped by default since mplot3d is VERY slow plotting 3d meshes
	# example_confocal3d()

	logging.info("Done.")
	plt.show()
