import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem,
    QVBoxLayout, QPushButton, QWidget, QFileDialog, QMenuBar, QAction, QHBoxLayout,
    QInputDialog, QLineEdit, QGraphicsEllipseItem, QTextEdit, QDialog, QLabel,
    QSlider, QColorDialog, QFormLayout, QComboBox, QMessageBox
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPen, QBrush, QCursor, QColor, QPolygonF, QPainter
)
from PyQt5.QtCore import Qt, QPointF, QPoint
import json
import os


class DuctSystemGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the default annotation names first
        self.default_annotation_names = [
            "CFP basal", "GFP basal", "YFP basal", "RFP basal",
            "CFP luminal", "GFP luminal", "YFP luminal", "RFP luminal"
        ]  # Default annotation names

        # Initialize annotation colors
        self.annotation_colors = self.generate_distinct_colors(len(self.default_annotation_names))

        self.annotation_point_size = 10  # Default point size
        self.annotation_line_thickness = 2  # Default line thickness

        # Initialize duct system
        self.duct_system = DuctSystem()
        self.current_point_name = None  # Current active point
        self.active_segment_name = None  # Current active segment for Segment Mode
        self.next_bp_name = 1  # Starting name for branch points
        self.intermediate_points = []  # List to store intermediate points for the current segment
        self.point_items = {}  # Dictionary to store point graphics items for easy access
        self.segment_items = {}  # Dictionary to store segment graphics items
        self.temp_line = None  # Temporary line for the current drawing segment
        self.dotted_lines = []  # Persistent dotted lines before the most recent branch point
        self.selection_mode = False  # Mode for selecting specific points
        self.segment_mode = False  # Mode for handling segments
        self.annotation_mode = None  # Mode for placing annotations
        self.panning_mode = False  # Mode for panning the view
        self.pan_start = QPoint()  # Starting position for panning
        self.custom_annotation_name = None  # Name for custom annotation mode

        # New features flags
        self.continuous_draw_mode = False  # Flag for continuous drawing
        self.drawing_continuous = False  # Flag to track continuous drawing
        self.annotate_region_mode = False  # Flag for region annotation mode
        self.new_origin_mode = False  # Flag for new origin mode

        # Initialize channels
        self.channels = {}  # Dictionary to store channel images per Z slice
        self.channel_brightness = {}  # Brightness settings for each channel
        self.current_z = 0  # Current Z slice index
        self.total_z_slices = 0  # Total number of Z slices

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Duct System Annotator")
        self.setGeometry(100, 100, 1600, 1000)  # Increased window size for better visibility

        # Menu bar setup
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        load_image_action = QAction('Load Image(s)', self)
        load_image_action.triggered.connect(self.load_tiff)
        file_menu.addAction(load_image_action)

        load_annotations_action = QAction('Load Annotations', self)
        load_annotations_action.triggered.connect(self.load_annotations)
        file_menu.addAction(load_annotations_action)

        save_annotations_action = QAction('Save Annotations', self)
        save_annotations_action.triggered.connect(self.save_annotations)
        file_menu.addAction(save_annotations_action)

        # Edit menu
        edit_menu = menubar.addMenu('Edit')

        edit_annotations_action = QAction('Edit Annotation Names', self)
        edit_annotations_action.triggered.connect(self.edit_annotation_names)
        edit_menu.addAction(edit_annotations_action)

        edit_properties_action = QAction('Edit Annotation Properties', self)
        edit_properties_action.triggered.connect(self.show_edit_properties_dialog)
        edit_menu.addAction(edit_properties_action)

        edit_line_color_action = QAction('Edit Line Colors', self)
        edit_line_color_action.triggered.connect(self.edit_line_colors)
        edit_menu.addAction(edit_line_color_action)

        # Modes menu
        modes_menu = menubar.addMenu('Modes')

        self.continuous_draw_action = QAction('Continuous Draw Mode', self, checkable=True)
        self.continuous_draw_action.triggered.connect(self.toggle_continuous_draw_mode)
        modes_menu.addAction(self.continuous_draw_action)

        annotate_segment_action = QAction('Annotate Segment', self)
        annotate_segment_action.triggered.connect(self.annotate_active_segment)
        edit_menu.addAction(annotate_segment_action)

        annotate_region_action = QAction('Annotate Region', self)
        annotate_region_action.triggered.connect(self.toggle_annotate_region_mode)
        modes_menu.addAction(annotate_region_action)

        # Channels menu
        channels_menu = menubar.addMenu('Channels')

        adjust_brightness_action = QAction('Adjust Brightness', self)
        adjust_brightness_action.triggered.connect(self.show_brightness_dialog)
        channels_menu.addAction(adjust_brightness_action)

        # Instructions menu
        instructions_menu = menubar.addMenu('Instructions')

        show_instructions_action = QAction('Show Instructions', self)
        show_instructions_action.triggered.connect(self.show_instructions_dialog)
        instructions_menu.addAction(show_instructions_action)

        # Initialize graphics scene and view
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)

        # Side buttons setup
        self.segment_mode_button = QPushButton("Segment Mode", self)
        self.segment_mode_button.setCheckable(True)
        self.segment_mode_button.clicked.connect(self.toggle_segment_mode)

        self.panning_mode_button = QPushButton("Panning Mode", self)
        self.panning_mode_button.setCheckable(True)
        self.panning_mode_button.clicked.connect(self.toggle_panning_mode)

        self.new_origin_button = QPushButton("New Origin", self)
        self.new_origin_button.clicked.connect(self.activate_new_origin_mode)

        self.z_prev_button = QPushButton("Previous Z Slice", self)
        self.z_prev_button.clicked.connect(self.prev_z_slice)

        self.z_next_button = QPushButton("Next Z Slice", self)
        self.z_next_button.clicked.connect(self.next_z_slice)

        self.z_label = QLabel("Z Slice: 0", self)

        self.annotation_buttons = []
        for name in self.default_annotation_names:
            button = QPushButton(name, self)
            button.clicked.connect(lambda _, n=name: self.activate_annotation_mode(n))
            button.setEnabled(False)  # Initially disabled
            self.annotation_buttons.append(button)

        # Add a "Specify Name" button
        specify_name_button = QPushButton("Specify Name", self)
        specify_name_button.clicked.connect(self.activate_specify_name_mode)
        specify_name_button.setEnabled(False)  # Initially disabled
        self.annotation_buttons.append(specify_name_button)

        # Layout setup
        side_layout = QVBoxLayout()
        side_layout.addWidget(self.segment_mode_button)
        side_layout.addWidget(self.panning_mode_button)
        side_layout.addWidget(self.new_origin_button)
        side_layout.addWidget(self.z_prev_button)
        side_layout.addWidget(self.z_next_button)
        side_layout.addWidget(self.z_label)
        for button in self.annotation_buttons:
            side_layout.addWidget(button)
        side_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addLayout(side_layout)
        main_layout.addWidget(self.view)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.view.setMouseTracking(True)
        self.view.mousePressEvent = self.handle_mouse_press
        self.view.mouseMoveEvent = self.handle_mouse_move
        self.view.mouseReleaseEvent = self.handle_mouse_release
        self.view.wheelEvent = self.handle_wheel_event  # Handle zooming

        # Set cursor to crosshair for clarity
        self.view.setCursor(QCursor(Qt.CrossCursor))

        self.update_mode_display()

    def load_annotations(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Annotations", "", "JSON Files (*.json);;All Files (*)")
        if file_name:
            with open(file_name, 'r') as file:
                data = json.load(file)
                # Clear only the annotations, keeping the image
                self.clear_annotations()

                # Load branch points
                for name, point in data.get('branch_points', {}).items():
                    point_qt = QPointF(point['x'], point['y'])
                    z = point.get('z', 0)  # Default Z slice is 0 if not provided
                    self.duct_system.add_branch_point(name, point_qt, z)
                    if z == self.current_z:
                        color = self.annotation_colors.get(name, Qt.green)
                        point_item = self.scene.addEllipse(
                            point_qt.x() - 5, point_qt.y() - 5,
                            self.annotation_point_size, self.annotation_point_size,
                            QPen(color), QBrush(color)
                        )
                        self.point_items[name] = point_item

                # Load segments
                for segment_name, segment in data.get('segments', {}).items():
                    start_bp = segment['start_bp']
                    end_bp = segment['end_bp']
                    intermediate_points = [(p['x'], p['y']) for p in segment['internal_points']]
                    start_z = segment.get('start_z', 0)
                    end_z = segment.get('end_z', 0)

                    self.duct_system.add_segment(start_bp, end_bp, segment_name, intermediate_points)
                    self.duct_system.segments[segment_name].set_z_coordinates(start_z, end_z)

                    if start_z == self.current_z or end_z == self.current_z:
                        self.draw_segment_with_intermediates(start_bp, end_bp, intermediate_points,
                                                             color_key=segment_name)

                    # Load annotations for the segment
                    for annotation in segment.get('annotations', []):
                        point = QPointF(annotation['x'], annotation['y'])
                        annotation_color = self.annotation_colors.get(annotation['name'], Qt.red)
                        self.scene.addEllipse(
                            point.x() - 5, point.y() - 5,
                            self.annotation_point_size, self.annotation_point_size,
                            QPen(annotation_color), QBrush(annotation_color)
                        )

                    # Load regions
                    for region_points in segment.get('regions', []):
                        polygon = QPolygonF([QPointF(p['x'], p['y']) for p in region_points])
                        self.scene.addPolygon(polygon, QPen(Qt.red, 2), QBrush(QColor(255, 0, 0, 50)))

            self.statusBar().showMessage("Annotations loaded successfully.")

    def save_annotations(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Annotations", "", "JSON Files (*.json);;All Files (*)")
        if file_name:
            data = {
                'branch_points': {
                    name: {'x': point['location'].x(), 'y': point['location'].y(), 'z': point['z']}
                    for name, point in self.duct_system.branch_points.items()
                },
                'segments': {
                    name: {
                        'start_bp': segment.start_bp,
                        'end_bp': segment.end_bp,
                        'internal_points': [{'x': p[0], 'y': p[1]} for p in segment.get_internal_points()],
                        'start_z': segment.start_z,
                        'end_z': segment.end_z,
                        'annotations': [{'name': a['name'], 'x': a['x'], 'y': a['y']} for a in segment.annotations],
                        'regions': [[{'x': p.x(), 'y': p.y()} for p in region] for region in segment.regions]
                    }
                    for name, segment in self.duct_system.segments.items()
                }
            }
            with open(file_name, 'w') as file:
                json.dump(data, file, indent=4)

            self.statusBar().showMessage("Annotations saved successfully.")

    def generate_distinct_colors(self, num_colors):
        """Generate a dictionary of distinct colors for annotations."""
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            color = QColor.fromHsv(int(hue * 360), 255, 255)
            colors.append(color)
        return {name: colors[i] for i, name in enumerate(self.default_annotation_names)}

    def load_tiff(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self, "Load TIFF Slices", "", "TIFF Files (*.tiff; *.tif);;All Files (*)",
            options=options
        )
        if files:
            # Assuming sorted order corresponds to Z slices
            sorted_files = sorted(files)
            # Load each Z slice for each channel
            self.channels = {}  # Reset channels
            for file in sorted_files:
                # Derive channel name from filename or prompt the user
                base = os.path.basename(file)
                channel_name, _ = QInputDialog.getText(
                    self, "Channel Name", f"Enter channel name for '{base}':",
                    QLineEdit.Normal, "Channel1"
                )
                if channel_name:
                    image = QImage(file)
                    if image.isNull():
                        QMessageBox.warning(self, "Load Image", f"Failed to load image: {file}")
                        continue
                    if channel_name not in self.channels:
                        self.channels[channel_name] = []
                        self.channel_brightness[channel_name] = 1.0  # Default brightness
                    self.channels[channel_name].append(QPixmap.fromImage(image))
            # Determine total Z slices based on the first channel
            if self.channels:
                self.total_z_slices = len(next(iter(self.channels.values())))
                self.current_z = 0
                self.display_current_z_slice()
                # Enable annotation buttons now that image is loaded
                for button in self.annotation_buttons:
                    button.setEnabled(True)
            else:
                QMessageBox.warning(self, "Load Image", "No valid images loaded.")

    def display_current_z_slice(self):
        if self.channels and self.total_z_slices > 0:
            # Combine channels with brightness settings
            combined_image = self.combine_channels()
            pixmap = QPixmap.fromImage(combined_image)
            if hasattr(self, 'pixmap_item') and self.pixmap_item:
                self.scene.removeItem(self.pixmap_item)
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.pixmap_item)
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self.z_label.setText(f"Z Slice: {self.current_z + 1}/{self.total_z_slices}")
            self.load_annotations_for_current_z()
        else:
            QMessageBox.warning(self, "Display Image", "No channels loaded.")

    def combine_channels(self):
        # Create a base image (black)
        first_channel = next(iter(self.channels))
        base_pixmap = self.channels[first_channel][self.current_z].copy()
        base_image = base_pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
        painter = QPainter(base_image)
        # Overlay other channels with brightness adjustments
        for channel, pixmaps in self.channels.items():
            if channel == first_channel:
                continue  # Skip the first channel as it's the base
            brightness = self.channel_brightness.get(channel, 1.0)
            adjusted_pixmap = self.adjust_brightness(pixmaps[self.current_z], brightness)
            painter.drawPixmap(0, 0, adjusted_pixmap)
        painter.end()
        return base_image

    def adjust_brightness(self, pixmap, brightness_factor):
        image = pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
        for y in range(image.height()):
            for x in range(image.width()):
                color = QColor(image.pixel(x, y))
                # Adjust brightness by scaling RGB values
                color.setRedF(min(color.redF() * brightness_factor, 1.0))
                color.setGreenF(min(color.greenF() * brightness_factor, 1.0))
                color.setBlueF(min(color.blueF() * brightness_factor, 1.0))
                image.setPixelColor(x, y, color)
        return QPixmap.fromImage(image)

    def prev_z_slice(self):
        if self.current_z > 0:
            self.current_z -= 1
            self.display_current_z_slice()
        else:
            QMessageBox.information(self, "Z Slice Navigation", "Already at the first Z slice.")

    def next_z_slice(self):
        if self.current_z < self.total_z_slices - 1:
            self.current_z += 1
            self.display_current_z_slice()
        else:
            QMessageBox.information(self, "Z Slice Navigation", "Already at the last Z slice.")

    def load_annotations_for_current_z(self):
        # Clear current annotations
        self.clear_annotations()

        # Load branch points and segments for the current Z slice
        for name, point in self.duct_system.branch_points.items():
            if point['z'] == self.current_z:
                point_qt = QPointF(point['location'].x(), point['location'].y())
                color = self.annotation_colors.get(point['name'], Qt.green)
                point_item = self.scene.addEllipse(
                    point_qt.x() - 5, point_qt.y() - 5,
                    self.annotation_point_size, self.annotation_point_size,
                    QPen(color), QBrush(color)
                )
                self.point_items[name] = point_item

        # Load segments and annotations for the current Z slice
        for segment_name, segment in self.duct_system.segments.items():
            if segment.start_z == self.current_z or segment.end_z == self.current_z:
                self.draw_segment_with_intermediates(
                    segment.start_bp, segment.end_bp,
                    list(segment.internal_points),
                    color_key=segment_name
                )
                # Load annotations
                for annotation in segment.annotations:
                    point = QPointF(annotation['x'], annotation['y'])
                    annotation_color = self.annotation_colors.get(annotation['name'], Qt.red)
                    self.scene.addEllipse(
                        point.x() - 5, point.y() - 5,
                        self.annotation_point_size, self.annotation_point_size,
                        QPen(annotation_color),
                        QBrush(annotation_color)
                    )
                # Load regions
                for region in segment.regions:
                    self.scene.addPolygon(
                        region,
                        QPen(Qt.red, 2),
                        QBrush(QColor(255, 0, 0, 50))
                    )

    def clear_annotations(self):
        # Remove all items except the image
        for item in self.scene.items():
            if isinstance(item, QGraphicsPixmapItem):
                continue
            self.scene.removeItem(item)
        self.point_items.clear()
        self.segment_items.clear()

    def handle_mouse_press(self, event):
        if self.panning_mode:
            self.pan_start = event.pos()  # Capture the mouse position at the start of the panning
            self.view.setCursor(Qt.ClosedHandCursor)
        elif self.annotate_region_mode:
            self.handle_region_mouse_press(event)
        else:
            point = self.view.mapToScene(event.pos())
            if self.annotation_mode and self.active_segment_name:
                self.add_annotation_point(point)
            elif self.selection_mode:
                self.clear_temp_line()  # Clear the temp line when selecting a new point
                self.clear_intermediate_points()  # Clear intermediate points on entering selection mode
                self.select_active_point(point)
                self.selection_mode = False  # Exit selection mode after selecting a point
            else:
                if event.button() == Qt.LeftButton:
                    if self.segment_mode:
                        self.handle_segment_selection(point)
                    elif self.new_origin_mode:
                        self.set_origin(point)
                        self.new_origin_mode = False
                        self.statusBar().showMessage("New origin set.")
                        self.view.setCursor(QCursor(Qt.ArrowCursor))
                    else:
                        self.handle_left_click(point)
                elif event.button() == Qt.RightButton:
                    if self.continuous_draw_mode:
                        self.drawing_continuous = True
                        self.add_intermediate_point(point)
                    elif not self.segment_mode and self.current_point_name is not None:
                        self.add_intermediate_point(point)

    def handle_mouse_move(self, event):
        if self.panning_mode and event.buttons() == Qt.LeftButton:
            # Handle panning
            delta = event.pos() - self.pan_start
            self.view.setTransformationAnchor(QGraphicsView.NoAnchor)
            self.view.setResizeAnchor(QGraphicsView.NoAnchor)
            self.view.horizontalScrollBar().setValue(
                self.view.horizontalScrollBar().value() - delta.x())
            self.view.verticalScrollBar().setValue(
                self.view.verticalScrollBar().value() - delta.y())
            self.pan_start = event.pos()
        elif self.segment_mode:
            # Clear the temp line if in segment mode but not actively drawing
            self.clear_temp_line()
        elif not self.selection_mode and not self.panning_mode and self.current_point_name is not None:
            # Update the temp line if no specific mode is active (normal mode)
            point = self.view.mapToScene(event.pos())
            self.update_temp_line(point)

        if self.continuous_draw_mode and self.drawing_continuous and event.buttons() & Qt.RightButton:
            point = self.view.mapToScene(event.pos())
            self.add_intermediate_point(point)

    def handle_mouse_release(self, event):
        if self.panning_mode:
            self.view.setCursor(Qt.OpenHandCursor)
        if self.continuous_draw_mode and event.button() == Qt.RightButton:
            self.drawing_continuous = False

    def handle_wheel_event(self, event):
        # Set the anchor to the center of the view for consistent zooming behavior
        self.view.setTransformationAnchor(QGraphicsView.AnchorViewCenter)

        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        # Zoom in or out based on the wheel movement
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        # Apply the zoom transformation, scaling the view
        self.view.scale(zoom_factor, zoom_factor)

    def handle_left_click(self, point):
        if self.current_point_name is None:
            self.add_branch_point(point)
        else:
            self.finalize_segment(point)

    def handle_segment_selection(self, point):
        # Check which segment (if any) the point is near
        for segment_name, segment_items in self.segment_items.items():
            for segment_item in segment_items:
                if segment_item.shape().contains(segment_item.mapFromScene(point)):
                    self.set_active_segment(segment_name)
                    return

    def add_branch_point(self, point, bp_name=None, z=None):
        if bp_name is None:
            bp_name = f"bp{self.next_bp_name}"
            self.next_bp_name += 1

        if z is None:
            z = self.current_z

        self.duct_system.add_branch_point(bp_name, point, z)
        color = self.annotation_colors.get(bp_name, Qt.green)
        point_item = self.scene.addEllipse(
            point.x() - 5, point.y() - 5,
            self.annotation_point_size, self.annotation_point_size,
            QPen(color), QBrush(color)
        )
        self.point_items[bp_name] = point_item
        self.set_active_point(bp_name)
        self.statusBar().showMessage(f"Branch point '{bp_name}' created at {point} on Z slice {z}.")

    def ensure_qpointf(self, point):
        if isinstance(point, tuple):
            return QPointF(point[0], point[1])
        return point  # If it's already a QPointF, return it as is

    def add_intermediate_point(self, point, color=Qt.gray):
        # Convert the last point to QPointF and ensure it's a copy
        if self.intermediate_points:
            last_point = self.ensure_qpointf(self.intermediate_points[-1])
        else:
            last_point = self.duct_system.get_branch_point(self.current_point_name)["location"]

        # Ensure the current point is a QPointF and store a copy as tuple
        point = self.ensure_qpointf(point)
        line = self.scene.addLine(
            last_point.x(), last_point.y(),
            point.x(), point.y(),
            QPen(color, 2)
        )
        self.dotted_lines.append(line)

        # Store the point as a copy
        self.intermediate_points.append((point.x(), point.y()))
        self.statusBar().showMessage(f"Intermediate point added at {point}.")

    def finalize_segment(self, end_point):
        bp_name = f"bp{self.next_bp_name}"
        z = self.current_z
        self.duct_system.add_branch_point(bp_name, end_point, z)

        segment_name = f"{self.current_point_name}to{bp_name}"
        self.duct_system.add_segment(
            self.current_point_name, bp_name, segment_name,
            list(self.intermediate_points)
        )
        self.duct_system.segments[segment_name].set_z_coordinates(
            self.duct_system.get_branch_point(self.current_point_name)["z"], z
        )

        self.draw_segment_with_intermediates(
            self.current_point_name, bp_name,
            list(self.intermediate_points),
            color_key=segment_name
        )

        color = self.annotation_colors.get(segment_name, Qt.blue)
        point_item = self.scene.addEllipse(
            end_point.x() - 5, end_point.y() - 5,
            self.annotation_point_size, self.annotation_point_size,
            QPen(color), QBrush(color)
        )
        self.point_items[bp_name] = point_item

        # Reset intermediate points and remove the temporary line
        self.intermediate_points.clear()
        self.clear_temp_line()
        self.clear_dotted_lines()  # Clear persistent dotted lines when finalizing the segment

        self.set_active_point(bp_name)
        self.next_bp_name += 1
        self.statusBar().showMessage(f"Segment '{segment_name}' created on Z slice {z}.")

    def update_temp_line(self, point):
        """Update the temporary line for the current segment."""
        self.clear_temp_line()

        start_point = self.duct_system.get_branch_point(self.current_point_name)["location"]

        # Draw temporary line from the last point (or start) to the current mouse position
        if self.intermediate_points:
            last_point = QPointF(*self.intermediate_points[-1])  # Convert back to QPointF
        else:
            last_point = start_point

        self.temp_line = self.scene.addLine(
            last_point.x(), last_point.y(),
            point.x(), point.y(),
            QPen(Qt.gray, 2, Qt.DashLine)
        )

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

    def clear_intermediate_points(self):
        """Clear all intermediate points."""
        self.intermediate_points.clear()

    def select_active_point(self, point):
        for bp_name, bp in self.duct_system.branch_points.items():
            if self.is_point_near(bp["location"], point):
                self.set_active_point(bp_name)
                return

    def set_active_point(self, bp_name):
        if self.current_point_name and self.current_point_name in self.point_items:
            old_active_item = self.point_items[self.current_point_name]
            old_active_item.setBrush(QBrush(Qt.green))

        if bp_name in self.point_items:
            self.current_point_name = bp_name
            active_item = self.point_items[bp_name]
            active_item.setBrush(QBrush(Qt.magenta))  # Mark active point
            self.statusBar().showMessage(f"Active point set to '{bp_name}'.")
        else:
            self.current_point_name = None  # Clear the current point if it doesn't exist
            self.statusBar().showMessage("No active point set.")

    def draw_segment_with_intermediates(self, start_bp_name, end_bp_name, intermediate_points, color_key=None):
        start_point = self.duct_system.get_branch_point(start_bp_name)["location"]
        end_point = self.duct_system.get_branch_point(end_bp_name)["location"]

        previous_point = start_point
        segment_lines = []  # List to store all lines for the segment

        # Determine color
        color = self.annotation_colors.get(color_key or "default_segment", Qt.blue)

        for point in intermediate_points:
            point_qt = QPointF(*point)  # Convert tuple back to QPointF
            line = self.scene.addLine(
                previous_point.x(), previous_point.y(),
                point_qt.x(), point_qt.y(),
                QPen(color, self.annotation_line_thickness)
            )
            segment_lines.append(line)
            previous_point = point_qt

        # Add the final line segment to complete the segment
        line = self.scene.addLine(
            previous_point.x(), previous_point.y(),
            end_point.x(), end_point.y(),
            QPen(color, self.annotation_line_thickness)
        )
        segment_lines.append(line)

        # Store the list of line items for the segment
        segment_name = f"{start_bp_name}to{end_bp_name}"
        self.segment_items[segment_name] = segment_lines

        # Set as active segment if in Segment Mode
        if self.segment_mode:
            self.set_active_segment(segment_name)

        # Handle regions if any
        segment = self.duct_system.get_segment(segment_name)
        if segment and segment.regions:
            for region in segment.regions:
                self.scene.addPolygon(
                    region,
                    QPen(Qt.red, 2),
                    QBrush(QColor(255, 0, 0, 50))
                )

    def set_active_segment(self, segment_name):
        # Reset the color of the previous active segment to its original color
        if self.active_segment_name and self.active_segment_name in self.segment_items:
            original_color = self.annotation_colors.get(self.active_segment_name, Qt.blue)
            for segment_item in self.segment_items[self.active_segment_name]:
                segment_item.setPen(QPen(original_color, self.annotation_line_thickness))

        # If segment_name is None, just clear the active segment without setting a new one
        if segment_name is None:
            self.active_segment_name = None
            return

        self.active_segment_name = segment_name

        # Highlight the new active segment in yellow
        for segment_item in self.segment_items[self.active_segment_name]:
            segment_item.setPen(QPen(Qt.yellow, self.annotation_line_thickness))

        self.statusBar().showMessage(f"Active segment set to '{segment_name}'.")

    def is_point_near(self, bp_location, click_point, threshold=10):
        return (bp_location - click_point).manhattanLength() < threshold

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_O:
            self.clear_temp_line()  # Clear the temp line when entering selection mode
            self.clear_dotted_lines()  # Clear all persistent dotted lines when entering selection mode
            self.clear_intermediate_points()  # Clear all intermediate points when entering selection mode
            self.selection_mode = True  # Enter selection mode
            self.segment_mode = False  # Exit segment mode
            self.panning_mode = False  # Exit panning mode

            # Reset annotation mode when switching modes
            self.annotation_mode = None
            self.custom_annotation_name = None

            self.update_mode_display()
            self.statusBar().showMessage("Selection mode: Click on a branch point to select it as the active point.")
        elif event.key() == Qt.Key_S:
            self.toggle_segment_mode()
        elif event.key() == Qt.Key_P:
            self.toggle_panning_mode()
        elif event.key() == Qt.Key_Backspace:
            self.revert_to_previous_branch_point()
        elif event.key() == Qt.Key_Delete:
            self.delete_most_recent_branch()
        elif event.key() == Qt.Key_N:
            self.activate_new_origin_mode()

    def toggle_segment_mode(self):
        self.segment_mode = not self.segment_mode
        self.selection_mode = False  # Exit selection mode
        self.panning_mode = False  # Exit panning mode

        if not self.segment_mode:
            self.annotation_mode = None
            self.custom_annotation_name = None
            self.set_active_segment(None)  # Reset active segment
            self.clear_temp_line()  # Clear the temporary line
            self.clear_dotted_lines()  # Clear all dotted lines

        self.update_mode_display()
        self.statusBar().showMessage("Segment mode activated." if self.segment_mode else "Segment mode deactivated.")

        for button in self.annotation_buttons:
            button.setEnabled(self.segment_mode)

        # Update cursor based on mode
        self.view.setCursor(QCursor(Qt.CrossCursor) if self.segment_mode else QCursor(Qt.ArrowCursor))

    def toggle_panning_mode(self):
        self.panning_mode = not self.panning_mode
        self.segment_mode = False  # Exit segment mode
        self.selection_mode = False  # Exit selection mode
        self.annotation_mode = None  # Exit annotation mode

        if self.panning_mode:
            self.clear_temp_line()  # Clear the temporary dotted line
            self.clear_dotted_lines()  # Clear all persistent dotted lines
            self.view.setCursor(Qt.OpenHandCursor)
            self.statusBar().showMessage("Panning mode activated. Drag to move the view, scroll to zoom.")
        else:
            self.view.setCursor(QCursor(Qt.ArrowCursor))
            self.statusBar().showMessage("Panning mode deactivated.")

        self.update_mode_display()

    def update_mode_display(self):
        mode = "Panning Mode" if self.panning_mode else "Selection Mode" if self.selection_mode else \
            "Segment Mode" if self.segment_mode else "Point Mode"
        self.segment_mode_button.setChecked(self.segment_mode)
        self.panning_mode_button.setChecked(self.panning_mode)
        self.z_prev_button.setEnabled(hasattr(self, 'channels') and self.total_z_slices > 0)
        self.z_next_button.setEnabled(hasattr(self, 'channels') and self.total_z_slices > 0)
        self.new_origin_button.setEnabled(True)  # Always enabled
        self.z_label.setEnabled(hasattr(self, 'channels') and self.total_z_slices > 0)
        self.statusBar().showMessage(f"Current Mode: {mode}")

    def activate_annotation_mode(self, name):
        if self.segment_mode and self.active_segment_name:
            self.annotation_mode = name
            self.custom_annotation_name = None  # Reset custom name
            self.statusBar().showMessage(f"Annotation mode: Click to place '{name}' annotations.")

    def activate_specify_name_mode(self):
        if self.segment_mode and self.active_segment_name:
            custom_name, ok = QInputDialog.getText(
                self, "Specify Annotation Name", "Enter annotation name:"
            )
            if ok and custom_name:
                self.annotation_mode = custom_name
                self.custom_annotation_name = custom_name
                self.statusBar().showMessage(f"Annotation mode: Click to place '{custom_name}' annotations.")
            else:
                self.annotation_mode = None
                self.custom_annotation_name = None  # Reset custom name if canceled

    def add_annotation_point(self, point):
        if self.annotation_mode and self.active_segment_name:
            annotation = {'name': self.annotation_mode, 'x': point.x(), 'y': point.y()}
            segment = self.duct_system.get_segment(self.active_segment_name)
            if segment:
                segment.add_annotation(annotation)
                # Use the color and size specific to the annotation type
                color = self.annotation_colors.get(self.annotation_mode, Qt.red)
                pen = QPen(color, self.annotation_line_thickness)
                brush = QBrush(color)
                self.scene.addEllipse(
                    point.x() - (self.annotation_point_size / 2),
                    point.y() - (self.annotation_point_size / 2),
                    self.annotation_point_size, self.annotation_point_size,
                    pen, brush
                )
                self.statusBar().showMessage(
                    f"Annotation '{self.annotation_mode}' added to segment '{segment.segment_name}'."
                )

    def revert_to_previous_branch_point(self):
        if self.current_point_name:
            # Only revert to previous branch points, not segment points
            previous_bp = self.duct_system.get_previous_branch_point(self.current_point_name)
            if previous_bp:
                self.set_active_point(previous_bp)
                self.statusBar().showMessage(f"Reverted to previous branch point '{previous_bp}'.")
            else:
                self.statusBar().showMessage("No previous branch point to revert to.")

    def edit_annotation_names(self):
        """Open a dialog to edit annotation button names."""
        for i, button in enumerate(self.annotation_buttons[:-1]):  # Exclude the "Specify Name" button
            new_name, ok = QInputDialog.getText(
                self, "Edit Annotation Name", f"Enter new name for '{button.text()}':"
            )
            if ok and new_name:
                self.default_annotation_names[i] = new_name
                button.setText(new_name)
                # Update the annotation_colors dictionary
                if new_name not in self.annotation_colors:
                    # Assign a new color if not already present
                    hue = i / len(self.default_annotation_names)
                    color = QColor.fromHsv(int(hue * 360), 255, 255)
                    self.annotation_colors[new_name] = color
                # Optionally, remove old name from annotation_colors
                # ...

    def show_edit_properties_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Annotation Properties")
        dialog.resize(400, 400)

        layout = QVBoxLayout()

        # Slider for point size
        layout.addWidget(QLabel("Point Size:"))
        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setMinimum(1)
        self.point_size_slider.setMaximum(20)
        self.point_size_slider.setValue(self.annotation_point_size)  # Reflect current value
        layout.addWidget(self.point_size_slider)

        # Slider for line thickness
        layout.addWidget(QLabel("Line Thickness:"))
        self.line_thickness_slider = QSlider(Qt.Horizontal)
        self.line_thickness_slider.setMinimum(1)
        self.line_thickness_slider.setMaximum(10)
        self.line_thickness_slider.setValue(self.annotation_line_thickness)  # Reflect current value
        layout.addWidget(self.line_thickness_slider)

        # Color pickers for each annotation type
        self.color_buttons = {}
        for annotation_name in self.default_annotation_names:
            layout.addWidget(QLabel(f"{annotation_name} Color:"))
            color_button = QPushButton("Choose Color")
            color_button.clicked.connect(lambda _, n=annotation_name: self.choose_annotation_color(n))
            layout.addWidget(color_button)
            self.color_buttons[annotation_name] = color_button

        # Apply and close buttons
        button_layout = QHBoxLayout()
        apply_button = QPushButton("Apply", dialog)
        apply_button.clicked.connect(self.apply_annotation_properties)
        button_layout.addWidget(apply_button)

        close_button = QPushButton("Close", dialog)
        close_button.clicked.connect(dialog.accept)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

        dialog.setLayout(layout)
        dialog.exec_()

    def choose_annotation_color(self, annotation_name):
        color = QColorDialog.getColor()
        if color.isValid():
            self.annotation_colors[annotation_name] = color

    def apply_annotation_properties(self):
        self.annotation_point_size = self.point_size_slider.value()
        self.annotation_line_thickness = self.line_thickness_slider.value()

        # Update the existing annotations' appearance
        for segment_name, segment in self.duct_system.segments.items():
            for annotation in segment.annotations:
                annotation_point = QPointF(annotation['x'], annotation['y'])
                items_at_point = self.scene.items(annotation_point)
                for item in items_at_point:
                    if isinstance(item, QGraphicsEllipseItem):
                        item.setRect(
                            annotation_point.x() - (self.annotation_point_size / 2),
                            annotation_point.y() - (self.annotation_point_size / 2),
                            self.annotation_point_size, self.annotation_point_size
                        )
                        item.setPen(QPen(self.annotation_colors.get(annotation['name'], Qt.red),
                                         self.annotation_line_thickness))
                        item.setBrush(QBrush(self.annotation_colors.get(annotation['name'], Qt.red)))

        # Update existing segment lines' appearance
        for segment_name, segment_lines in self.segment_items.items():
            for line_item in segment_lines:
                line_item.setPen(QPen(Qt.blue, self.annotation_line_thickness))

        self.statusBar().showMessage("Annotation properties updated.")

    def edit_line_colors(self):
        if not self.active_segment_name:
            self.statusBar().showMessage("No active segment selected.")
            return

        color = QColorDialog.getColor()
        if color.isValid():
            # Update the color in the annotation_colors dictionary
            self.annotation_colors[self.active_segment_name] = color

            # Update the visual representation
            if self.active_segment_name in self.segment_items:
                for line_item in self.segment_items[self.active_segment_name]:
                    line_item.setPen(QPen(color, self.annotation_line_thickness))

            self.statusBar().showMessage(f"Segment '{self.active_segment_name}' color updated.")

    def annotate_active_segment(self):
        if not self.active_segment_name:
            self.statusBar().showMessage("No active segment to annotate.")
            return

        segment = self.duct_system.get_segment(self.active_segment_name)
        if not segment:
            self.statusBar().showMessage("Active segment not found.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Annotate Segment: {self.active_segment_name}")
        dialog.resize(300, 200)

        layout = QVBoxLayout()

        # Example properties: Type, Description
        layout.addWidget(QLabel("Type:"))
        self.segment_type_input = QLineEdit(dialog)
        layout.addWidget(self.segment_type_input)

        layout.addWidget(QLabel("Description:"))
        self.segment_description_input = QLineEdit(dialog)
        layout.addWidget(self.segment_description_input)

        # Add more property inputs as needed

        # Apply and Close buttons
        button_layout = QHBoxLayout()
        apply_button = QPushButton("Apply", dialog)
        apply_button.clicked.connect(lambda: self.apply_segment_properties(dialog, segment))
        button_layout.addWidget(apply_button)

        close_button = QPushButton("Close", dialog)
        close_button.clicked.connect(dialog.accept)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        dialog.exec_()

    def apply_segment_properties(self, dialog, segment):
        segment_type = self.segment_type_input.text()
        description = self.segment_description_input.text()

        if segment_type:
            segment.add_property("Type", segment_type)
        if description:
            segment.add_property("Description", description)

        self.statusBar().showMessage(f"Properties applied to segment '{segment.segment_name}'.")
        dialog.accept()

    def toggle_continuous_draw_mode(self):
        self.continuous_draw_mode = not self.continuous_draw_mode
        self.statusBar().showMessage("Continuous Draw Mode "
                                     f"{'Activated' if self.continuous_draw_mode else 'Deactivated'}.")

    def toggle_annotate_region_mode(self):
        self.annotate_region_mode = not self.annotate_region_mode
        if self.annotate_region_mode:
            self.region_points = []  # Start a new region
            self.statusBar().showMessage("Region Annotation Mode: Click to define vertices. Right-click to complete.")
            self.view.setCursor(QCursor(Qt.CrossCursor))
        else:
            self.statusBar().showMessage("Region Annotation Mode deactivated.")
            self.view.setCursor(QCursor(Qt.ArrowCursor))

    def handle_region_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            point = self.view.mapToScene(event.pos())
            self.region_points.append(point)
            if len(self.region_points) > 1:
                polygon = self.create_polygon(self.region_points)
                if hasattr(self, 'region_polygon'):
                    self.scene.removeItem(self.region_polygon)
                self.region_polygon = self.scene.addPolygon(polygon, QPen(Qt.red, 2), QBrush(QColor(255, 0, 0, 50)))
        elif event.button() == Qt.RightButton and len(self.region_points) > 2:
            # Finalize region and associate with the active segment
            polygon = self.create_polygon(self.region_points)
            if self.active_segment_name:
                segment = self.duct_system.get_segment(self.active_segment_name)
                if segment:
                    segment.add_region(polygon)
                    self.statusBar().showMessage(f"Region added to segment '{self.active_segment_name}'.")
            self.region_points.clear()
            self.toggle_annotate_region_mode()

    def create_polygon(self, points):
        return QPolygonF(points)

    def activate_new_origin_mode(self):
        self.new_origin_mode = True
        self.statusBar().showMessage("New Origin Mode: Click to set a new origin.")
        self.view.setCursor(QCursor(Qt.CrossCursor))

    def set_origin(self, point):
        # Set a new branch point as the origin
        origin_name = f"origin_{self.next_bp_name}"
        self.add_branch_point(point, origin_name)
        self.statusBar().showMessage(f"New origin set at {point}.")

    def show_brightness_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Adjust Channel Brightness")
        dialog.resize(300, 400)

        layout = QVBoxLayout()

        self.brightness_sliders = {}
        for channel in self.channels.keys():
            layout.addWidget(QLabel(f"{channel} Brightness:"))
            slider = QSlider(Qt.Horizontal, dialog)
            slider.setMinimum(1)
            slider.setMaximum(200)  # 1.0 to 2.0 brightness
            slider.setValue(int(self.channel_brightness[channel] * 100))
            slider.setObjectName(channel)
            slider.valueChanged.connect(self.update_channel_brightness)
            layout.addWidget(slider)
            self.brightness_sliders[channel] = slider

        close_button = QPushButton("Close", dialog)
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def update_channel_brightness(self):
        slider = self.sender()
        if slider:
            channel = slider.objectName()
            brightness = slider.value() / 100.0
            self.channel_brightness[channel] = brightness
            self.display_current_z_slice()

    def show_instructions_dialog(self):
        instructions = (
            "Instructions:\n\n"
            "- Click 'Load Image(s)' to load TIFF images (Z slices).\n"
            "- Use 'Segment Mode' to draw segments.\n"
            "- Use 'Panning Mode' to move the view, scroll to zoom.\n"
            "- Use the 'Edit' menu to customize annotation names.\n"
            "- Press 'S' to toggle Segment Mode, 'P' for Panning Mode, and 'N' for New Origin."
            "\n\n"
            "Developed for annotating biological ducts with Z-slice navigation."
        )

        dialog = QDialog(self)
        dialog.setWindowTitle("Instructions")
        dialog.resize(400, 300)

        layout = QVBoxLayout()

        text_edit = QTextEdit(dialog)
        text_edit.setReadOnly(True)
        text_edit.setText(instructions)
        layout.addWidget(text_edit)

        close_button = QPushButton("Close", dialog)
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.exec_()


# Supporting classes for Duct System (unchanged)
class DuctSystem:
    def __init__(self):
        self.branch_points = {}
        self.segments = {}

    def add_branch_point(self, name, location, z):
        self.branch_points[name] = {"name": name, "location": location, "z": z}

    def get_branch_point(self, name):
        return self.branch_points.get(name)

    def get_previous_branch_point(self, name):
        # Helper method to find the previous branch point
        sorted_names = sorted(self.branch_points.keys())
        try:
            index = sorted_names.index(name)
            if index > 0:
                return sorted_names[index - 1]
        except ValueError:
            return None

    def remove_branch_point(self, name):
        if name in self.branch_points:
            del self.branch_points[name]

    def add_segment(self, start_bp, end_bp, segment_name, intermediate_points):
        if start_bp in self.branch_points and end_bp in self.branch_points:
            segment = DuctSegment(start_bp, end_bp, segment_name)
            segment.internal_points = intermediate_points  # Store as tuples
            self.segments[segment_name] = segment

    def get_segment(self, segment_name):
        return self.segments.get(segment_name)


class DuctSegment:
    def __init__(self, start_bp, end_bp, segment_name):
        self.start_bp = start_bp
        self.end_bp = end_bp
        self.segment_name = segment_name
        self.internal_points = []  # List to store internal points between branch points
        self.annotations = []  # List to store annotations
        self.regions = []  # List to store regions (polygons)
        self.start_z = None  # Z-coordinate of start branch point
        self.end_z = None  # Z-coordinate of end branch point
        self.properties = {}  # Dictionary to store segment properties

    def set_z_coordinates(self, start_z, end_z):
        self.start_z = start_z
        self.end_z = end_z

    def add_internal_point(self, point):
        """Add an internal point to the segment."""
        self.internal_points.append((point.x(), point.y()))  # Store as tuple

    def get_internal_points(self):
        return self.internal_points

    def add_annotation(self, annotation):
        """Add an annotation to the segment."""
        self.annotations.append(annotation)

    def add_region(self, polygon):
        """Add a polygon region to the segment."""
        self.regions.append(polygon)

    def add_property(self, key, value):
        self.properties[key] = value

    def get_property(self, key):
        return self.properties.get(key)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DuctSystemGUI()
    ex.show()
    sys.exit(app.exec_())
