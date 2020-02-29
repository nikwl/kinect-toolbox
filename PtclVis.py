from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import sys

from .Kinect import Kinect

class Visualizer(object):
    def __init__(self, roi=False):
        #QT app
        self.app = QtGui.QApplication([])
        self.w = gl.GLViewWidget()
        self.w.setWindowTitle('pyqtgraph')
        self.w.show()
        g = gl.GLGridItem()
        self.w.addItem(g)

        #initialize some points data
        pos = np.zeros((1,3))

        self.scatter_plot = gl.GLScatterPlotItem(pos=pos)
        self.w.addItem(self.scatter_plot)

        self.k = Kinect()
        
        self.zone = None
        if roi:
            self.zone = self.k.get_roi()

        self.animation()

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def update(self):
        ptcld = self.k.get_ptcld(self.zone)
        self.scatter_plot.setData(pos=ptcld, size=2)

    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(20)
        self.start()

# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    v = Visualizer()