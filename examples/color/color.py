import pykokkos as pk

import argparse
import copy
import importlib
import math
import random
import sys

import numpy as np

import kernels 
from kernels import *

"""
class Parameters:
    def __init__(self):
        self.algorithm = 0
        self.repeat = 6
        self.chunk_size = -1
        self.shmemsize = 16128
        self.verbose_level = 0
        self.check_output = 0
        self.coloring_input_file = None
        self.coloring_output_file = None
        self.output_histogram = 0
        self.output_graphviz = 0
        self.output_graphviz_vert_max = 1500
        self.use_threads = 0
        self.use_openmp = 0
        self.use_cuda = 0
        self.use_serial = 0
        self.validate = 0
        self.mtx_bin_file = None
"""

Ordinal = "default_lno_t"
Offset = "default_size_type"
Layout = "default_layout"
ExecSpace = "Kokkos::DefaultExecutionSpace"
DeviceSpace = ExecSpace + "::memory_space"

Handle = KokkosKernelsHandle(Offset, Ordinal, "default_scalar", ExecSpace, DeviceSpace, DeviceSpace)

gridX = 15
gridY = 25
numVertices = gridX * gridY

def getVertexID(x, y):
    return y * gridX + x


def getVertexPos(vert):
    return (int(vert % gridX), int(vert / gridX))


def printColoring(colors, numColors):
    # skip mirrorview
    colorsHost = colors
    numDigits = math.ceil(math.log10(numColors + 1))
    numFmt = f"%{numDigits + 1}d"
    for y in range(gridY):
        for x in range(gridX):
            vertex = getVertexID(x, y)
            color = colorsHost[vertex]
            print(numFmt % color, end='')
        print()


def generate9pt():
    rowmap =  np.zeros(numVertices + 1, dtype=np.int32)
    colinds = []

    rowmap[0] = 0
    for vert in range(numVertices):
        x, y = getVertexPos(vert)
        for ny in range(y-1, y+2):
            for nx in range(x-1, x+2):
                if nx == x and ny == y:
                    continue
                if nx < 0 or nx >= gridX or ny < 0 or ny >= gridY:
                    continue
                colinds.append(getVertexID(nx, ny))

        rowmap[vert+1] = len(colinds)

    numEdges = len(colinds)
    rowmapHost: pk.View1D = pk.View([numVertices + 1], pk.int32, space=pk.HostSpace)
    colindsHost: pk.View1D = pk.View([numEdges], pk.int32, space=pk.HostSpace)

    # TODO init view from numpy array
    np.copyto(rowmapHost.data, rowmap)
    np.copyto(colindsHost.data, np.array(colinds))

    rowmapDevice = pk.View([numVertices + 1], pk.int32)
    colindsDevice = pk.View([numEdges], pk.int32)

    deep_copy(rowmapDevice, rowmapHost)
    deep_copy(colindsDevice, colindsHost)

    return rowmapDevice, colindsDevice


def run() -> None:  
    global gridX, gridY, numVertices
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, 
            help="determines number of elem in x direction (default: 15)")
    parser.add_argument('-M', type=int, 
            help="determines number of elem in y direction (default: 25)")
    parser.add_argument('--cuda', action="store_true", help="use CUDA (default: 0)")
    parser.add_argument('--file', type=str, help="output timing info to file") 

    args = parser.parse_args()

    if args.N:
        gridX = args.N
    if args.M:
        gridY = args.M
    numVertices = gridX * gridY

    if args.cuda:
        pk.enable_uvm()
        pk.set_default_space(pk.Cuda)

    timer = pk.Timer()
    rowmapDevice, colindsDevice = generate9pt()

    init_time = timer.seconds()
    timer.reset()

    # run distance-1 coloring
    handle = Handle()
    handle.create_graph_coloring_handle()
 
    graph_color(ptr(handle), numVertices, numVertices, rowmapDevice, colindsDevice)
    colorHandle = handle.get_graph_coloring_handle()

    colors = colorHandle.get_vertex_colors()
    numColors = colorHandle.get_num_colors()
    print("9-pt stencil: Distance-1 Colors (used %d): " % (numColors))
    # printColoring(colors, numColors)
    print()
    handle.destroy_graph_coloring_handle()

    # run distance-2 coloring
    handle = Handle()
    handle.create_distance2_graph_coloring_handle()

    graph_color_distance2(ptr(handle), numVertices, rowmapDevice, colindsDevice)
    colorHandleD2 = handle.get_distance2_graph_coloring_handle()

    colors = colorHandleD2.get_vertex_colors()
    numColors = colorHandleD2.get_num_colors()
    print("9-pt stencil: Distance-2 Colors (used %d): " % (numColors))
    # printColoring(colors, numColors)
    print()
    handle.destroy_distance2_graph_coloring_handle()

    time = timer.seconds()
    # outupt timing info
    if args.file:
        with open(args.file, "a") as f:
            # workload, cuda, size1, size2, time
            f.write("%s, %d, %d, %d, %f, %f\n" % ("color", args.cuda, gridX, gridY, init_time, time))


if __name__ == "__main__":
    run()
