import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QVBoxLayout, QPushButton, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage, QPen, QBrush, QPainter, QMouseEvent
from PyQt5.QtCore import Qt, QPointF


class DuctSystemGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.duct_system = DuctSystem()
        self.current_point_name = None  # Current active point
        self.next_bp_name = 1  # Starting name for branch points
        self.intermediate_points = []  # List to store intermediate points for the current segment
        self.point_items = {}  # Dictionary to store point graphics items for easy access
        self.temp_line = None  # Temporary line for the current drawing segment
        self.dotted_lines = []  # Persistent dotted lines before the most recent branch point
        self.selection_mode = False  # Mode for selecting specific points
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Duct System Annotator")
        self.setGeometry(100, 100, 1200, 900)  # Larger window for better visibility

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)

        self.load_button = QPushButton("Load TIFF", self)
        self.load_button.clicked.connect(self.load_tiff)

        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(self.clear_scene)

        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.view)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.view.setMouseTracking(True)
        self.view.mousePressEvent = self.handle_mouse_press
        self.view.mouseMoveEvent = self.handle_mouse_move

        # Enable zooming and panning
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setInteractive(True)
        self.view.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)

    def load_tiff(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load TIFF", "", "TIFF Files (*.tiff; *.tif);;All Files (*)", options=options)
        if file_name:
            image = QImage(file_name)
            pixmap = QPixmap.fromImage(image)
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.pixmap_item)
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def clear_scene(self):
        self.scene.clear()
        self.duct_system = DuctSystem()
        self.current_point_name = None
        self.next_bp_name = 1
        self.intermediate_points.clear()
        self.point_items.clear()
        self.temp_line = None
        self.dotted_lines.clear()

    def handle_mouse_press(self, event: QMouseEvent):
        point = self.view.mapToScene(event.pos())

        if self.selection_mode:
            self.clear_temp_line()  # Clear the temp line when selecting a new point
            self.select_active_point(point)
            self.selection_mode = False  # Exit selection mode after selecting a point
        else:
            if event.button() == Qt.LeftButton:
                self.handle_left_click(point)
            elif event.button() == Qt.RightButton:
                self.handle_right_click(point)

    def handle_mouse_move(self, event: QMouseEvent):
        if self.current_point_name is not None and not self.selection_mode:
            point = self.view.mapToScene(event.pos())
            self.update_temp_line(point)

    def handle_left_click(self, point):
        if self.current_point_name is None:
            self.add_branch_point(point)
        else:
            self.finalize_segment(point)

    def handle_right_click(self, point):
        if self.current_point_name is not None:
            self.add_intermediate_point(point)

    def add_branch_point(self, point):
        bp_name = str(self.next_bp_name)
        self.duct_system.add_branch_point(bp_name, point)
        point_item = self.scene.addEllipse(point.x() - 5, point.y() - 5, 10, 10, QPen(Qt.green), QBrush(Qt.green))
        self.point_items[bp_name] = point_item
        self.set_active_point(bp_name)
        self.next_bp_name += 1
        self.statusBar().showMessage(f"Branch point '{bp_name}' created at {point}.")

    def add_intermediate_point(self, point):
        # Draw a persistent dotted line for each intermediate point added
        if self.intermediate_points:
            last_point = self.intermediate_points[-1]
        else:
            last_point = self.duct_system.get_branch_point(self.current_point_name)["location"]

        dotted_line = self.scene.addLine(last_point.x(), last_point.y(), point.x(), point.y(), QPen(Qt.gray, 2, Qt.DashLine))
        self.dotted_lines.append(dotted_line)

        self.intermediate_points.append(point)
        self.statusBar().showMessage(f"Intermediate point added at {point}.")

    def finalize_segment(self, end_point):
        bp_name = str(self.next_bp_name)
        self.duct_system.add_branch_point(bp_name, end_point)

        segment_name = f"{self.current_point_name}to{bp_name}"
        self.duct_system.add_segment(self.current_point_name, bp_name, segment_name, self.intermediate_points)

        self.draw_segment_with_intermediates(self.current_point_name, bp_name, self.intermediate_points)

        point_item = self.scene.addEllipse(end_point.x() - 5, end_point.y() - 5, 10, 10, QPen(Qt.red), QBrush(Qt.red))
        self.point_items[bp_name] = point_item

        # Reset intermediate points and remove the temporary line
        self.intermediate_points.clear()
        self.clear_temp_line()
        self.clear_dotted_lines()  # Clear persistent dotted lines when finalizing the segment

        self.set_active_point(bp_name)
        self.next_bp_name += 1
        self.statusBar().showMessage(f"Segment '{segment_name}' created.")

    def update_temp_line(self, point):
        """Update the temporary line for the current segment."""
        self.clear_temp_line()

        start_point = self.duct_system.get_branch_point(self.current_point_name)["location"]

        # Draw temporary line from the last point (or start) to the current mouse position
        if self.intermediate_points:
            last_point = self.intermediate_points[-1]
        else:
            last_point = start_point

        self.temp_line = self.scene.addLine(last_point.x(), last_point.y(), point.x(), point.y(), QPen(Qt.gray, 2, Qt.DashLine))

    def clear_temp_line(self):
        """Clear the current temporary line."""
        if self.temp_line is not None:
            self.scene.removeItem(self.temp_line)
            self.temp_line = None

    def clear_dotted_lines(self):
        """Clear all persistent dotted lines."""
        for line in self.dotted_lines:
            self.scene.removeItem(line)
        self.dotted_lines.clear()

    def select_active_point(self, point):
        for bp_name, bp in self.duct_system.branch_points.items():
            if self.is_point_near(bp["location"], point):
                self.set_active_point(bp_name)
                return

    def set_active_point(self, bp_name):
        if self.current_point_name and self.current_point_name in self.point_items:
            old_active_item = self.point_items[self.current_point_name]
            old_active_item.setBrush(QBrush(Qt.green))

        self.current_point_name = bp_name
        active_item = self.point_items[bp_name]
        active_item.setBrush(QBrush(Qt.magenta))  # Mark active point
        self.statusBar().showMessage(f"Active point set to '{bp_name}'.")

    def draw_segment_with_intermediates(self, start_bp_name, end_bp_name, intermediate_points):
        start_point = self.duct_system.get_branch_point(start_bp_name)["location"]
        end_point = self.duct_system.get_branch_point(end_bp_name)["location"]

        previous_point = start_point
        for point in intermediate_points:
            self.scene.addLine(previous_point.x(), previous_point.y(), point.x(), point.y(), QPen(Qt.blue, 2))
            previous_point = point

        self.scene.addLine(previous_point.x(), previous_point.y(), end_point.x(), end_point.y(), QPen(Qt.blue, 2))

    def is_point_near(self, bp_location, click_point, threshold=10):
        return (bp_location - click_point).manhattanLength() < threshold

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Backspace:
            self.revert_to_previous_branch_point()
        elif event.key() == Qt.Key_O:
            self.clear_temp_line()  # Clear the temp line when entering selection mode
            self.clear_dotted_lines()  # Clear all persistent dotted lines when entering selection mode
            self.selection_mode = True  # Enter selection mode
            self.statusBar().showMessage("Selection mode: Click on a branch point to select it as the active point.")

    def revert_to_previous_branch_point(self):
        if self.current_point_name:
            # Only revert to previous branch points, not segment points
            for segment_name, segment in reversed(self.duct_system.segments.items()):
                if segment.end_bp == self.current_point_name:
                    self.set_active_point(segment.start_bp)
                    self.statusBar().showMessage(f"Reverted to previous branch point '{self.current_point_name}'.")
                    break


class DuctSystem:
    def __init__(self):
        self.branch_points = {}  # Dictionary of branch points
        self.segments = {}  # Dictionary of segments

    def add_branch_point(self, name, location):
        self.branch_points[name] = {"name": name, "location": location}

    def get_branch_point(self, name):
        return self.branch_points.get(name)

    def add_segment(self, start_bp, end_bp, segment_name, intermediate_points):
        if start_bp in self.branch_points and end_bp in self.branch_points:
            segment = DuctSegment(start_bp, end_bp, segment_name)
            segment.internal_points = intermediate_points  # Store intermediate points in the segment
            self.segments[segment_name] = segment
        else:
            print(f"One or both branch points '{start_bp}', '{end_bp}' do not exist.")

    def get_segment(self, segment_name):
        return self.segments.get(segment_name)


class DuctSegment:
    def __init__(self, start_bp, end_bp, segment_name):
        """
        Initialize a duct segment.
        :param start_bp: The starting branch point (name or ID).
        :param end_bp: The ending branch point (name or ID).
        :param segment_name: The name of the segment (e.g., "bp1tobp2").
        """
        self.start_bp = start_bp
        self.end_bp = end_bp
        self.segment_name = segment_name
        self.internal_points = []  # List to store internal points between the branch points

    def add_internal_point(self, point):
        """Add an internal point to the segment."""
        self.internal_points.append(point)

    def get_internal_points(self):
        """Return the list of internal points."""
        return self.internal_points


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DuctSystemGUI()
    ex.show()
    sys.exit(app.exec_())
