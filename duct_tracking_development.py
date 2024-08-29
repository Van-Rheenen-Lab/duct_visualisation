import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QVBoxLayout, \
    QPushButton, QWidget, QFileDialog, QMenuBar, QAction, QHBoxLayout, QInputDialog, QLineEdit, QGraphicsEllipseItem ,QTextEdit, QDialog, QLabel, QSlider, QColorDialog
from PyQt5.QtGui import QPixmap, QImage, QPen, QBrush, QCursor, QColor
from PyQt5.QtCore import Qt, QPointF, QPoint
import json

class DuctSystemGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the default annotation names first
        self.default_annotation_names = ["CFP basal", "GFP basal", "YFP basal", "RFP basal", "CFP luminal", "GFP luminal", "YFP luminal", "RFP luminal"]  # Default annotation names

        # Now you can initialize the annotation colors using the default names
        self.annotation_colors = {name: Qt.red for name in self.default_annotation_names}  # Default colors
        self.annotation_point_size = 10  # Default point size
        self.annotation_line_thickness = 2  # Default line thickness

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
        self.initUI()


    def initUI(self):
        self.setWindowTitle("Duct System Annotator")
        self.setGeometry(100, 100, 1200, 900)  # Larger window for better visibility

        # Menu bar setup
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        load_image_action = QAction('Load Image', self)
        load_image_action.triggered.connect(self.load_tiff)
        file_menu.addAction(load_image_action)

        load_annotations_action = QAction('Load Annotations', self)
        load_annotations_action.triggered.connect(self.load_annotations)
        file_menu.addAction(load_annotations_action)

        save_annotations_action = QAction('Save Annotations', self)
        save_annotations_action.triggered.connect(self.save_annotations)
        file_menu.addAction(save_annotations_action)

        # Add menu for editing annotation names and properties
        edit_menu = menubar.addMenu('Edit')

        edit_annotations_action = QAction('Edit Annotation Names', self)
        edit_annotations_action.triggered.connect(self.edit_annotation_names)
        edit_menu.addAction(edit_annotations_action)

        edit_properties_action = QAction('Edit Annotation Properties', self)
        edit_properties_action.triggered.connect(self.show_edit_properties_dialog)
        edit_menu.addAction(edit_properties_action)

        # Add Instructions menu
        instructions_menu = menubar.addMenu('Instructions')

        show_instructions_action = QAction('Show Instructions', self)
        show_instructions_action.triggered.connect(self.show_instructions_dialog)
        instructions_menu.addAction(show_instructions_action)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)

        # Side buttons setup
        self.segment_mode_button = QPushButton("Segment Mode", self)
        self.segment_mode_button.setCheckable(True)
        self.segment_mode_button.clicked.connect(self.toggle_segment_mode)

        self.panning_mode_button = QPushButton("Panning Mode", self)
        self.panning_mode_button.setCheckable(True)
        self.panning_mode_button.clicked.connect(self.toggle_panning_mode)

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

    def load_tiff(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load TIFF", "", "TIFF Files (*.tiff; *.tif);;All Files (*)",
                                                   options=options)
        if file_name:
            image = QImage(file_name)
            pixmap = QPixmap.fromImage(image)
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.pixmap_item)
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def load_annotations(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Annotations", "", "JSON Files (*.json);;All Files (*)")
        if file_name:
            with open(file_name, 'r') as file:
                data = json.load(file)
                # Clear only the annotations, keeping the image
                self.clear_scene(clear_image=False)

                # Load branch points and store them correctly
                highest_bp_name = 0
                for name, point in data.get('branch_points', {}).items():
                    point_qt = QPointF(point['x'], point['y'])
                    self.duct_system.add_branch_point(name, point_qt)
                    point_item = self.scene.addEllipse(point_qt.x() - 5, point_qt.y() - 5, 10, 10, QPen(Qt.green),
                                                       QBrush(Qt.green))
                    self.point_items[name] = point_item  # Store the point item in the dictionary
                    highest_bp_name = max(highest_bp_name, int(name))  # Track the highest branch point name

                # Update next_bp_name based on the highest branch point name loaded
                self.next_bp_name = highest_bp_name + 1

                # Load segments and re-link to branch points
                for segment_name, segment in data.get('segments', {}).items():
                    start_bp = segment['start_bp']
                    end_bp = segment['end_bp']
                    intermediate_points = [(p['x'], p['y']) for p in segment['internal_points']]

                    self.duct_system.add_segment(start_bp, end_bp, segment_name, intermediate_points)
                    self.draw_segment_with_intermediates(
                        start_bp,
                        end_bp,
                        intermediate_points,
                        color=Qt.blue
                    )
                    # Load annotations
                    for annotation in segment.get('annotations', []):
                        point = QPointF(annotation['x'], annotation['y'])
                        annotation_color = self.annotation_colors.get(annotation['name'], Qt.red)
                        self.scene.addEllipse(point.x() - 5, point.y() - 5, 10, 10, QPen(annotation_color),
                                              QBrush(annotation_color))

                # Set the current_point_name to the last branch point loaded
                if highest_bp_name > 0:
                    self.set_active_point(str(highest_bp_name))

    def show_edit_properties_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Annotation Properties")
        dialog.resize(400, 300)

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

        # Update the color only if the user has selected a new one
        if hasattr(self, 'selected_annotation_color'):
            self.annotation_color = self.selected_annotation_color

        self.statusBar().showMessage("Annotation properties updated.")

    def show_instructions_dialog(self):
        instructions = (
            "Instructions:\n\n"
            "- Click 'Load Image' to load a TIFF image.\n"
            "- Use 'Segment Mode' to draw segments.\n"
            "- Use 'Panning Mode' to move the view, Scrolling always works for zooming.\n"
            "- Press 'S' to toggle Segment Mode.\n"
            "- Press 'P' to toggle Panning Mode.\n"
            "- Press 'O' to select branch points.\n"
            "- Press 'Delete' to remove the last branch.\n"
            "- Press 'Backspace' to revert the active point to the previous point.\n"
            "- Use the 'Edit' menu to customize annotation names. \n\n"
            
            "Made by Jeroen Doornbos based on a version from Jacco van Rheenen\n"

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

    def save_annotations(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Annotations", "", "JSON Files (*.json);;All Files (*)")
        if file_name:
            data = {
                'branch_points': {name: {'x': point['location'].x(), 'y': point['location'].y()}
                                  for name, point in self.duct_system.branch_points.items()},
                'segments': {
                    name: {
                        'start_bp': segment.start_bp,
                        'end_bp': segment.end_bp,
                        'internal_points': [{'x': p[0], 'y': p[1]} for p in segment.get_internal_points()],
                        'annotations': [{'name': a['name'], 'x': a['x'], 'y': a['y']} for a in segment.annotations]
                    }
                    for name, segment in self.duct_system.segments.items()
                }
            }
            with open(file_name, 'w') as file:
                json.dump(data, file)

    def clear_scene(self, clear_image=True):
        """Clear the scene, keeping the image if specified."""
        if clear_image:
            self.scene.clear()
            self.pixmap_item = None  # Clear the image item if specified
        else:
            # Only remove items that are not the image
            for item in self.scene.items():
                if isinstance(item, QGraphicsPixmapItem):
                    continue  # Skip the image item
                self.scene.removeItem(item)

        self.duct_system = DuctSystem()
        self.current_point_name = None
        self.active_segment_name = None
        self.next_bp_name = 1
        self.intermediate_points.clear()
        self.point_items.clear()
        self.segment_items.clear()
        self.temp_line = None
        self.dotted_lines.clear()

    def handle_mouse_press(self, event):
        if self.panning_mode:
            self.pan_start = event.pos()  # Capture the mouse position at the start of the panning
            self.view.setCursor(Qt.ClosedHandCursor)
        else:
            point = self.view.mapToScene(event.pos())
            if self.annotation_mode:
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
                    else:
                        self.handle_left_click(point)
                elif event.button() == Qt.RightButton:
                    if not self.segment_mode and self.current_point_name is not None:
                        self.add_intermediate_point(point)

    def handle_mouse_move(self, event):
        if self.panning_mode and event.buttons() == Qt.LeftButton:
            # Calculate the difference in mouse movement
            delta = event.pos() - self.pan_start
            self.view.setTransformationAnchor(QGraphicsView.NoAnchor)
            self.view.setResizeAnchor(QGraphicsView.NoAnchor)
            self.view.horizontalScrollBar().setValue(self.view.horizontalScrollBar().value() - delta.x())
            self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().value() - delta.y())
            # Update pan_start to the new mouse position
            self.pan_start = event.pos()
        elif self.current_point_name is not None and not self.selection_mode and not self.segment_mode:
            point = self.view.mapToScene(event.pos())
            self.update_temp_line(point)

    def handle_mouse_release(self, event):
        if self.panning_mode:
            self.view.setCursor(Qt.OpenHandCursor)

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

    def add_branch_point(self, point, bp_name=None):
        if bp_name is None:
            bp_name = str(self.next_bp_name)
            self.next_bp_name += 1

        self.duct_system.add_branch_point(bp_name, point)
        point_item = self.scene.addEllipse(point.x() - 5, point.y() - 5, 10, 10, QPen(Qt.green), QBrush(Qt.green))
        self.point_items[bp_name] = point_item
        self.set_active_point(bp_name)
        self.statusBar().showMessage(f"Branch point '{bp_name}' created at {point}.")

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
        line = self.scene.addLine(last_point.x(), last_point.y(), point.x(), point.y(), QPen(color, 2))
        self.dotted_lines.append(line)

        # Store the point as a copy
        self.intermediate_points.append((point.x(), point.y()))
        self.statusBar().showMessage(f"Intermediate point added at {point}.")

    def finalize_segment(self, end_point):
        bp_name = str(self.next_bp_name)
        self.duct_system.add_branch_point(bp_name, end_point)

        segment_name = f"{self.current_point_name}to{bp_name}"
        self.duct_system.add_segment(self.current_point_name, bp_name, segment_name,
                                     list(self.intermediate_points))  # Copy list

        self.draw_segment_with_intermediates(self.current_point_name, bp_name,
                                             list(self.intermediate_points))  # Copy list

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
            last_point = QPointF(*self.intermediate_points[-1])  # Convert back to QPointF
        else:
            last_point = start_point

        self.temp_line = self.scene.addLine(last_point.x(), last_point.y(), point.x(), point.y(),
                                            QPen(Qt.gray, 2, Qt.DashLine))

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

        self.current_point_name = bp_name
        active_item = self.point_items[bp_name]
        active_item.setBrush(QBrush(Qt.magenta))  # Mark active point
        self.statusBar().showMessage(f"Active point set to '{bp_name}'.")

    def draw_segment_with_intermediates(self, start_bp_name, end_bp_name, intermediate_points, color=Qt.blue):
        start_point = self.duct_system.get_branch_point(start_bp_name)["location"]
        end_point = self.duct_system.get_branch_point(end_bp_name)["location"]

        previous_point = start_point
        segment_lines = []  # List to store all lines for the segment

        for point in intermediate_points:
            point_qt = QPointF(*point)  # Convert tuple back to QPointF
            line = self.scene.addLine(previous_point.x(), previous_point.y(), point_qt.x(), point_qt.y(),
                                      QPen(color, 2))
            segment_lines.append(line)
            previous_point = point_qt

        # Add the final line segment to complete the segment
        line = self.scene.addLine(previous_point.x(), previous_point.y(), end_point.x(), end_point.y(), QPen(color, 2))
        segment_lines.append(line)

        # Store the list of line items for the segment
        segment_name = f"{start_bp_name}to{end_bp_name}"
        self.segment_items[segment_name] = segment_lines

        # Set as active segment if in Segment Mode
        if self.segment_mode:
            self.set_active_segment(segment_name)

    def set_active_segment(self, segment_name):
        # Reset the color of the previous active segment to blue
        if self.active_segment_name and self.active_segment_name in self.segment_items:
            for segment_item in self.segment_items[self.active_segment_name]:
                segment_item.setPen(QPen(Qt.blue, 2))

        # If segment_name is None, just clear the active segment without setting a new one
        if segment_name is None:
            self.active_segment_name = None
            return

        self.active_segment_name = segment_name

        # Highlight the new active segment in yellow
        for segment_item in self.segment_items[self.active_segment_name]:
            segment_item.setPen(QPen(Qt.yellow, 2))

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
            self.delete_most_recent_branch()  # Call the delete method when Delete is pressed

    def toggle_segment_mode(self):
        self.segment_mode = not self.segment_mode
        self.selection_mode = False  # Exit selection mode
        self.panning_mode = False  # Exit panning mode

        # Reset annotation mode when segment mode is deactivated
        if not self.segment_mode:
            self.annotation_mode = None
            self.custom_annotation_name = None
            self.set_active_segment(None)  # Reset active segment to None and reset its color

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
        self.update_mode_display()

        if self.panning_mode:
            self.view.setCursor(Qt.OpenHandCursor)
            self.statusBar().showMessage("Panning mode activated. Drag to move the view, scroll to zoom.")
        else:
            self.view.setCursor(Qt.ArrowCursor)
            self.statusBar().showMessage("Panning mode deactivated.")

    def update_mode_display(self):
        mode = "Panning Mode" if self.panning_mode else "Selection Mode" if self.selection_mode else \
            "Segment Mode" if self.segment_mode else "Point Mode"
        self.segment_mode_button.setChecked(self.segment_mode)
        self.panning_mode_button.setChecked(self.panning_mode)
        self.statusBar().showMessage(f"Current Mode: {mode}")

    def activate_annotation_mode(self, name):
        if self.segment_mode and self.active_segment_name:
            self.annotation_mode = name
            self.custom_annotation_name = None  # Reset custom name
            self.statusBar().showMessage(f"Annotation mode: Click to place '{name}' annotations.")

    def activate_specify_name_mode(self):
        if self.segment_mode and self.active_segment_name:
            custom_name, ok = QInputDialog.getText(self, "Specify Annotation Name", "Enter annotation name:")
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
                # Use the color specific to the annotation type
                color = self.annotation_colors.get(self.annotation_mode, Qt.red)
                pen = QPen(color, self.annotation_line_thickness)
                brush = QBrush(color)
                self.scene.addEllipse(point.x() - (self.annotation_point_size / 2),
                                      point.y() - (self.annotation_point_size / 2),
                                      self.annotation_point_size, self.annotation_point_size, pen, brush)
                self.statusBar().showMessage(
                    f"Annotation '{self.annotation_mode}' added to segment '{segment.segment_name}'.")

    def revert_to_previous_branch_point(self):
        if self.current_point_name:
            # Only revert to previous branch points, not segment points
            for segment_name, segment in reversed(self.duct_system.segments.items()):
                if segment.end_bp == self.current_point_name:
                    self.set_active_point(segment.start_bp)
                    self.statusBar().showMessage(f"Reverted to previous branch point '{segment.start_bp}'.")
                    break

    def edit_annotation_names(self):
        """Open a dialog to edit annotation button names."""
        for i, button in enumerate(self.annotation_buttons[:-1]):  # Exclude the "Specify Name" button
            new_name, ok = QInputDialog.getText(self, "Edit Annotation Name", f"Enter new name for '{button.text()}':")
            if ok and new_name:
                self.default_annotation_names[i] = new_name
                button.setText(new_name)

    def delete_most_recent_branch(self):
        if not self.duct_system.branch_points:
            self.statusBar().showMessage("No branch points to delete.")
            return

        # Identify the most recent branch point
        last_bp_name = str(self.next_bp_name - 1)

        # Check if the branch point exists
        if last_bp_name not in self.duct_system.branch_points:
            self.statusBar().showMessage("No branch point found to delete.")
            return

        # Remove associated segments
        segments_to_remove = []
        for segment_name, segment in self.duct_system.segments.items():
            if segment.start_bp == last_bp_name or segment.end_bp == last_bp_name:
                segments_to_remove.append(segment_name)

        for segment_name in segments_to_remove:
            self.remove_segment(segment_name)

        # Remove the branch point from the duct_system and scene
        self.duct_system.remove_branch_point(last_bp_name)
        if last_bp_name in self.point_items:
            self.scene.removeItem(self.point_items[last_bp_name])
            del self.point_items[last_bp_name]

        # Update the next branch point name
        self.next_bp_name -= 1

        # Update current point name
        if self.next_bp_name > 1:
            self.set_active_point(str(self.next_bp_name - 1))
        else:
            self.current_point_name = None

        self.statusBar().showMessage(f"Branch point '{last_bp_name}' and associated segments deleted.")

    def remove_segment(self, segment_name):
        if segment_name in self.segment_items:
            # Remove the visual segment lines
            for segment_item in self.segment_items[segment_name]:
                self.scene.removeItem(segment_item)
            del self.segment_items[segment_name]

        if segment_name in self.duct_system.segments:
            # Remove annotations associated with this segment
            segment = self.duct_system.segments[segment_name]
            for annotation in segment.annotations:
                annotation_point = QPointF(annotation['x'], annotation['y'])
                items_at_point = self.scene.items(annotation_point)
                for item in items_at_point:
                    if isinstance(item, QGraphicsEllipseItem) and item.pen().color() == Qt.red:
                        self.scene.removeItem(item)
            del self.duct_system.segments[segment_name]


class DuctSystem:
    def __init__(self):
        self.branch_points = {}  # Dictionary of branch points
        self.segments = {}  # Dictionary of segments

    def add_branch_point(self, name, location):
        self.branch_points[name] = {"name": name, "location": location}

    def get_branch_point(self, name):
        return self.branch_points.get(name)

    def remove_branch_point(self, name):
        if name in self.branch_points:
            del self.branch_points[name]

    def add_segment(self, start_bp, end_bp, segment_name, intermediate_points):
        if start_bp in self.branch_points and end_bp in self.branch_points:
            segment = DuctSegment(start_bp, end_bp, segment_name)
            segment.internal_points = intermediate_points  # Store as tuples
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
        self.annotations = []  # List to store annotations

    def add_internal_point(self, point):
        """Add an internal point to the segment."""
        self.internal_points.append((point.x(), point.y()))  # Store as tuple

    def get_internal_points(self):
        """Return the list of internal points."""
        return self.internal_points

    def add_annotation(self, annotation):
        """Add an annotation to the segment."""
        self.annotations.append(annotation)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DuctSystemGUI()
    ex.show()
    sys.exit(app.exec_())
