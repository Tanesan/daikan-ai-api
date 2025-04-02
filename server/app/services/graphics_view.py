from PyQt6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,  QPushButton, QFileDialog, QMessageBox, QVBoxLayout, QHBoxLayout, QWidget
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush, QMouseEvent
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtCore import qDebug, QRectF
import io
import cv2
from requests import session
from zoom_handler import ZoomHandler


class GraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setMouseTracking(True)
        #enable rubber band selection
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        #set background color to black
        self.setBackgroundBrush(QBrush(Qt.GlobalColor.lightGray))
        #set the selection rubber band background color to half transparent blue after
        self.points = [] # list of QPointF. Each point is a point in the scene
        self.point_items = [] # list of QGraphicsEllipseItem. Each item is a point in self.points

    def wheelEvent(self, event):
        self.zoom_handler.handleWheelEvent(event, super().wheelEvent)
    
    def set_zoom_handler(self, zoom_handler):
        self.zoom_handler = zoom_handler
        
    def set_graphics_scene(self, graphics_scene):
        self.graphics_scene = graphics_scene
        
    def mousePressEvent(self, event: QMouseEvent):
        button = event.button()
        keys = event.modifiers()
        if button == Qt.MouseButton.LeftButton:
            if keys == Qt.KeyboardModifier.ControlModifier:
                self.add_point(event.pos())

        super().mousePressEvent(event)

    def add_point(self, mouse_point):
        point = self.mapToScene(mouse_point)
        self.points.append(point)
        self.draw_point(point)

    def draw_point(self, point):
        brush = QBrush(Qt.GlobalColor.red)
        self.graphics_scene.addEllipse(point.x() - 2.5, point.y() - 2.5, 5, 5, QPen(Qt.GlobalColor.red), brush)
    
    def clear_points(self):
        for item in self.point_items:
            self.graphics_scene.removeItem(item)
        self.points.clear()
        self.point_items.clear()
    
    def get_points(self):
        return self.points
    
    def get_pixmap_item(self):
        for item in self.graphics_scene.items():
            if isinstance(item, QGraphicsPixmapItem):
                return item
        return None

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            # Get the rubber band rect in view coordinates
            rubberBandRect = self.rubberBandRect()

            # Map corners of the rubber band rect to the scene to consider transformations
            topLeft = self.mapToScene(rubberBandRect.topLeft())
            bottomRight = self.mapToScene(rubberBandRect.bottomRight())

            # Create a QRectF from the mapped points
            sceneRect = QRectF(topLeft, bottomRight)
            if hasattr(self, 'selected_area_item') and self.selected_area_item in self.scene().items():
                self.scene().removeItem(self.selected_area_item)
            self.selected_area_item = self.scene().addRect(sceneRect, QPen(Qt.GlobalColor.darkBlue))
            #set a background color to the item and set transparenct toi half of it
            self.selected_area_item.setBrush(QBrush(QColor(0, 0, 255, 128)))
            # Now, sceneRect should correctly represent the rubber band selection in the scene

            self.selected_area = self.mapToImage(sceneRect)
            print(self.selected_area)
        super().mouseReleaseEvent(event)

    def mapToImage(self, sceneRect):
        #Get first pixmap items from the scene
        pixmap_item = self.get_pixmap_item()
        if pixmap_item is None:
            print ("No pixmap item found")
            return QRectF()

        # Get the transformation matrix of the pixmapItem
        transform = pixmap_item.transform()

        # Extract scale factors
        scaleX = transform.m11()  # Horizontal scale
        scaleY = transform.m22()  # Vertical scale

        # Calculate the bounding rect of the pixmap item to understand its position in the scene
        pixmapRect = pixmap_item.boundingRect()

        # Adjust sceneRect position based on pixmapItem position and scale
        adjustedRect = QRectF(
            (sceneRect.x() - pixmapRect.x()) / scaleX,
            (sceneRect.y() - pixmapRect.y()) / scaleY,
            sceneRect.width() / scaleX,
            sceneRect.height() / scaleY
        )

        return adjustedRect

    def mapPointToImage(self, point):
        #Get first pixmap items from the scene
        pixmap_item = self.get_pixmap_item()
        if pixmap_item is None:
            print ("No pixmap item found")
            return QPointF()
        
        # Get the transformation matrix of the pixmap_item
        transform = pixmap_item.transform()

        # Extract scale factors
        scaleX = transform.m11()  # Horizontal scale
        scaleY = transform.m22()
        
        # Calculate the bounding rect of the pixmap item to understand its position in the scene
        pixmapRect = pixmap_item.boundingRect()
        
        # Adjust sceneRect position based on pixmap_item position and scale
        adjustedPoint = QPointF(
            (point.x() - pixmapRect.x()) / scaleX,
            (point.y() - pixmapRect.y()) / scaleY
        )
        
        return adjustedPoint