#!/usr/bin/env python3

import sys

from PyQt5.QtGui import (
        QOpenGLVersionProfile,
        QOpenGLVertexArrayObject,
        QSurfaceFormat,
        QIcon
    )

from PyQt5.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QOpenGLWidget,
        QPushButton,
        QAction
    )

from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtCore import Qt

import pyVFRendering as vfr


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.widget = glWidget(self)
        self.setCentralWidget(self.widget)
        self.setWindowTitle("pyVFRendering Demo")

        self.initUI()

    def initUI(self):
        # Menu bar
        menu = self.menuBar().addMenu("File")
        action = menu.addAction("Take Screenshot")
        action.triggered.connect(self.takeScreenshot)
        # Status bar
        self.statusBar().showMessage("Ready", 5000)

    def takeScreenshot(self):
        filename = "screenshot"
        self.widget.takeScreenshot(filename)
        self.statusBar().showMessage("Saved screenshot to file '" + filename + ".bmp'", 5000)


class glWidget(QOpenGLWidget):

    def __init__(self, parent):
        QOpenGLWidget.__init__(self, parent)
        self.previous_mouse_position = [0,0]

    def takeScreenshot(self, filename):
        pixmap = self.grab()
        pixmap.save(filename + ".bmp")

    def initializeGL(self):
        # Geometry
        n_cells = [20, 20, 20]
        geometry = vfr.Geometry.cartesianGeometry(n_cells, [-20, -20, -20], [20, 20, 20])
        
        # Directions
        directions = []
        for iz in range(n_cells[2]):
            for iy in range(n_cells[1]):
                for ix in range(n_cells[0]):
                    directions.append( [0, 0, 1] )
        directions[0] = [1,0,0] # for colormap test

        # View
        self.view = vfr.View()
        self.vf   = vfr.VectorField(geometry, directions)
        
        # ArrowRenderer
        renderer_arrows = vfr.ArrowRenderer(self.view, self.vf)
        # BoundingBoxRenderer
        renderer_boundingbox = vfr.BoundingBoxRenderer.forCuboid(self.view, (geometry.min() + geometry.max())*0.5, [40,40,40], [0, 0, 0], 1)
        # CoordinateSystemRenderer
        renderer_cs = vfr.CoordinateSystemRenderer(self.view)
        renderer_cs.setAxisLength([20, 20, 20])
        renderer_cs.setNormalize(True)
        # Add renderers to view
        self.renderers = [ (renderer_arrows, [0.0, 0.0, 1.0, 1.0]), (renderer_boundingbox, [0.0, 0.0, 1.0, 1.0]), (renderer_cs, [0.0, 0.0, 0.2, 0.2]) ]
        self.view.renderers(self.renderers, True)

        # Options
        self.options = vfr.Options()
        self.options.setSystemCenter((geometry.min() + geometry.max())*0.5)
        self.options.setCameraPosition([0, -80, 20])
        self.options.setCenterPosition([0, 0, 0])
        self.options.setUpVector([0, 0, 1])
        self.options.setColormapImplementation(vfr.getColormapImplementation(vfr.Colormap.hsv))
        self.view.updateOptions(self.options)

    def paintGL(self):
        self.view.draw()

    def resizeGL(self, width, height):
        self.view.setFramebufferSize(width*self.window().devicePixelRatio(), height*self.window().devicePixelRatio())
        QTimer.singleShot(1, self.update)

    def mousePressEvent(self, event):
        self.previous_mouse_position = [event.x(), event.y()]

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            camera_mode = vfr.CameraMovementModes.rotate_bounded
        elif event.buttons() == Qt.RightButton:
            camera_mode = vfr.CameraMovementModes.translate
        else:
            return
        
        current_mouse_position = [event.x(), event.y()]
        self.view.mouseMove(self.previous_mouse_position, current_mouse_position, camera_mode)
        self.previous_mouse_position = current_mouse_position
        
        QTimer.singleShot(1, self.update)
    
    def wheelEvent(self, event):
        wheel_delta = event.angleDelta().y()
        self.view.mouseScroll(-wheel_delta * 0.1)
        
        QTimer.singleShot(1, self.update)


if __name__ == '__main__':
    # Set default surface format for OpenGL context
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)

    # Open the Application Window
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(640, 480)
    window.show()

    # Return
    sys.exit(app.exec_())