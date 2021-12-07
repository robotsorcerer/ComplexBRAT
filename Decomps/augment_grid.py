__all__ = ["augmentGrid"]

__author__ 		= "Lekan Molu"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import copy
import logging
import cupy as cp
import numpy as np
from LevelSetPy.Utilities import *
logger = logging.getLogger(__name__)

def augmentGrid(gridIn, dim, width=None):
    """
        Augment grid so that the top and bottom of the mesh steps
        have extrapolated mesh step sizes that allows us to efficiently 
        compute the upwinding differences.

        Lekan Molux, December 6, 2021
    """

    if not width:
        width = 1

    if((width < 0) or (width > gridIn.shape[dim])):
        error('Illegal width parameter')
    
    slopeMultiplier = +1

    gridOut = copy.deepcopy(gridIn)
    
    # create cell array with array size
    dims = gridIn.dim
    sizeIn = list(gridIn.shape) 
    indicesOut = []
    for i in range(dims):
        indicesOut.append(cp.arange(sizeIn[i], dtype=cp.intp))
    indicesIn = copy.copy(indicesOut)

    # create appropriately sized output array
    sizeOut = copy.copy(sizeIn)
    sizeOut[dim] = sizeOut[dim] + (2 * width)
    gridOut.dx[dim] = cp.zeros(tuple(sizeOut), dtype=cp.float64)

    indicesOut[dim] = cp.arange(width, sizeOut[dim] - width, dtype=cp.intp) # correct
    gridOut.dx[dim][cp.ix_(*indicesOut)] = copy.copy(gridIn.dx[dim]) # correct

    # compute slopes
    indicesOut[dim] = [0]
    indicesIn[dim] = [1]
    slopeBot = gridIn.dx[dim][cp.ix_(*indicesOut)] - gridIn.dx[dim][cp.ix_(*indicesIn)]

    indicesOut[dim] = [sizeIn[dim]-1]
    indicesIn[dim] = [sizeIn[dim] - 2]
    slopeTop = gridIn.dx[dim][cp.ix_(*indicesOut)] - gridIn.dx[dim][cp.ix_(*indicesIn)]
    #logger.debug(f'slopeBot: {cp.linalg.norm(slopeBot, 2)} slopeTop: {cp.linalg.norm(slopeTop, 2)}')

    # adjust slope sign to correspond with sign of data at array edge
    indicesIn[dim] = [0]
    slopeBot = slopeMultiplier * cp.abs(slopeBot) * cp.sign(gridIn.dx[dim][cp.ix_(*indicesIn)])

    indicesIn[dim] = [sizeIn[dim]-1]
    slopeTop = slopeMultiplier * cp.abs(slopeTop) * cp.sign(gridIn.dx[dim][cp.ix_(*indicesIn)])

    # now extrapolate
    for i in range(width):
        indicesOut[dim] = [i]
        indicesIn[dim] = [0]
        gridOut.dx[dim][cp.ix_(*indicesOut)] = (gridIn.dx[dim][cp.ix_(*indicesIn)] + (width - i) * slopeBot)

        indicesOut[dim] = [sizeOut[dim]-1-i]
        indicesIn[dim] = [sizeIn[dim]-1]
        gridOut.dx[dim][cp.ix_(*indicesOut)] = (gridIn.dx[dim][cp.ix_(*indicesIn)] + (width - i) * slopeTop)
    
    return gridOut