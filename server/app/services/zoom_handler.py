from PyQt6.QtCore import QObject, pyqtSignal, QPointF
from PyQt6.QtWidgets import QGraphicsView, QGraphicsItem, QScrollBar
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtCore import Qt

from PyQt6.QtCore import pyqtSlot

class ZoomHandler(QObject):
    zoomChanged = pyqtSignal(float)

    MINIMUM_SCALE_FACTOR = 0.000001
    MAXIMUM_SCALE_FACTOR = 10000000

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.view = view
        self.scaleFactor = 1.0
        self.setZoomScale(0.03)

    def handleWheelEvent(self, event, callbackDefaultViewBehavior):
        self.lastScenePosition = self.view.mapToScene(event.position().toPoint())
        self.lastMousePosition = event.position()
        zoomPercentage = 1.2 if event.angleDelta().y() > 0 else 0.8
        self.adjustZoomByScaleFactor(zoomPercentage)

    def adjustZoomByScaleFactor(self, factor):
        targetScaleFactor = self.scaleFactor * factor
        targetScaleFactor = max(min(targetScaleFactor, self.MAXIMUM_SCALE_FACTOR), self.MINIMUM_SCALE_FACTOR)
        self.setZoomScale(targetScaleFactor)

    def setZoomScale(self, newScale):
        if newScale == self.scaleFactor:
            return
        relativeScale = newScale / self.scaleFactor
        self.view.scale(relativeScale, relativeScale)
        self.scaleFactor = newScale
        if hasattr(self, 'lastMousePosition'):
            self.adjustScrollBars(self.lastMousePosition)
        self.zoomChanged.emit(self.scaleFactor)

    def adjustScrollBars(self, lastMousePosition):
        p1mouse = self.view.mapFromScene(self.lastScenePosition)
        move = QPointF(p1mouse.x(), p1mouse.y()) - lastMousePosition
        self.view.horizontalScrollBar().setValue(int(move.x()) + self.view.horizontalScrollBar().value())
        self.view.verticalScrollBar().setValue(int(move.y()) + self.view.verticalScrollBar().value())

    def zoomFitInView(self, rect, padding=0.0):
        view = self.view
        viewRect = view.viewport().rect()
        scaleFactor = min((viewRect.width() - padding) / rect.width(), (viewRect.height() - padding) / rect.height())
        scaleMin = 0.001
        scaleFactor = max(scaleFactor, scaleMin)
        view.resetTransform()
        view.scale(scaleFactor, scaleFactor)
        rw = scaleFactor * rect.width()
        rh = scaleFactor * rect.height()
        diffX = viewRect.width() - rw
        diffY = viewRect.height() - rh
        view.ensureVisible(rect, int(diffX * 0.5), int(diffY * 0.5))

    def zoomExtents(self, items, padding=0.0):
        view = self.view
        items = view.scene().items()
        if items:
            boundingRect = items[0].sceneBoundingRect()
            for item in items[1:]:
                boundingRect = boundingRect.united(item.sceneBoundingRect())
            self.zoomFitInView(boundingRect, padding)
        else:
            view.resetTransform()
            horizontalMiddle = (view.horizontalScrollBar().minimum() + view.horizontalScrollBar().maximum()) / 2
            verticalMiddle = (view.verticalScrollBar().minimum() + view.verticalScrollBar().maximum()) / 2
            view.horizontalScrollBar().setValue(horizontalMiddle)
            view.verticalScrollBar().setValue(verticalMiddle)

    def centerInView(self, items):
        view = self.view
        tm = view.transform()
        view.resetTransform()
        if items:
            boundingRect = items[0].sceneBoundingRect()
            for item in items[1:]:
                boundingRect = boundingRect.united(item.sceneBoundingRect())
            targetPoint = view.mapFromScene(boundingRect.center())
            viewCenter = view.mapToScene(view.viewport().rect().center())
            move = view.mapFromScene(viewCenter - targetPoint.toPointF())
            view.horizontalScrollBar().setValue(view.horizontalScrollBar().value() - move.x())
            view.verticalScrollBar().setValue(view.verticalScrollBar().value() - move.y())
        else:
            horizontalMiddle = (view.horizontalScrollBar().minimum() + view.horizontalScrollBar().maximum()) / 2
            verticalMiddle = (view.verticalScrollBar().minimum() + view.verticalScrollBar().maximum()) / 2
            view.horizontalScrollBar().setValue(int(horizontalMiddle))
            view.verticalScrollBar().setValue(int(verticalMiddle))
        view.setTransform(tm)

    @pyqtSlot()
    def zoom_to_items(self):
        boundingRect = self.view.scene().itemsBoundingRect()
        if boundingRect.isValid():
            self.view.fitInView(boundingRect, Qt.AspectRatioMode.KeepAspectRatio)