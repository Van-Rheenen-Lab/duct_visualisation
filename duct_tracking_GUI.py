import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem,
    QVBoxLayout, QPushButton, QWidget, QFileDialog, QAction, QHBoxLayout,
    QInputDialog, QGraphicsEllipseItem, QTextEdit, QDialog, QLabel,
    QSlider, QColorDialog, QMessageBox, QDoubleSpinBox, QGraphicsPolygonItem, QListWidget, QListWidgetItem
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPen, QBrush, QCursor, QColor, QPolygonF
)
from PyQt5.QtCore import Qt, QPointF, QPoint, QTimer, QEvent
import json
import numpy as np
import tifffile
import cv2

class DuctSystemGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.annotation_point_size = 10  # Default point size
        self.annotation_line_thickness = 2  # Default line thickness

        # Initialize duct systems
        self.duct_systems = []  # List to hold all duct systems
        self.active_duct_system = DuctSystem()
        self.duct_systems.append(self.active_duct_system)
        self.current_point_name = None  # Current active point
        self.active_segment_name = None  # Current active segment for Segment Mode
        self.next_bp_name = 1  # Starting name for branch points
        self.intermediate_points = []  # List to store intermediate points for the current segment
        self.point_items = {}  # Dictionary to store point graphics items for easy access
        self.segment_items = {}  # Dictionary to store segment graphics items
        self.temp_line = None  # Temporary line for the current drawing segment
        self.dotted_lines = []  # Persistent dotted lines before the most recent branch point
        self.selection_mode = False  # Mode for selecting specific points
        self.annotation_mode = None  # Mode for placing annotations
        self.panning_mode = False  # Mode for panning the view
        self.pan_start = QPoint()  # Starting position for panning
        self.custom_annotation_name = None  # Name for custom annotation mode
        self.annotation_colors = {}

        self.segment_annotations = {}  # Dictionary to store annotation names and their keybindings
        self.active_segment_annotation = None  # Currently active annotation
        self.config_file = "config.json"  # Config file to save/load settings
        self.load_config()  # Load settings at startup

        self.outlines_data = []
        self.outline_items = []
        self.outline_color = Qt.red  # Default outline color

        self.continuous_draw_mode = False  # Flag for continuous drawing
        self.drawing_continuous = False  # Flag to track continuous drawing
        self.new_origin_mode = False  # Flag for new origin mode

        # Initialize channels
        self.channels = {}  # Dictionary to store channel images per Z slice
        self.channel_brightness = {}  # Brightness settings for each channel
        self.current_z = 0  # Current Z slice index
        self.total_z_slices = 0  # Total number of Z slices
        self.scale_factor = 1  # Default downscale factor

        self.installEventFilter(self)  # Install an event filter to capture key presses

        # Timer for resetting the status bar message after showing key bindings info
        self.status_message_timer = QTimer(self)
        self.status_message_timer.setInterval(2000)  # 2 seconds
        self.status_message_timer.timeout.connect(self.clear_status_message)

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

        load_outlines_action = QAction('Load Outlines', self)
        load_outlines_action.triggered.connect(self.load_outlines)
        file_menu.addAction(load_outlines_action)

        # Edit menu
        edit_menu = menubar.addMenu('Edit')

        configure_segment_annotations_action = QAction('Configure Segment Annotations', self)
        configure_segment_annotations_action.triggered.connect(self.configure_segment_annotations)
        edit_menu.addAction(configure_segment_annotations_action)

        change_outline_color_action = QAction('Change Outline Color', self)
        change_outline_color_action.triggered.connect(self.change_outline_color)
        edit_menu.addAction(change_outline_color_action)

        delete_selected_bp_action = QAction('Delete Selected Branch Point And All Children', self)
        delete_selected_bp_action.triggered.connect(self.delete_selected_branch_point_and_descendants)
        edit_menu.addAction(delete_selected_bp_action)

        edit_line_color_action = QAction('Edit Default Line Color', self)
        edit_line_color_action.triggered.connect(self.edit_line_colors)
        edit_menu.addAction(edit_line_color_action)

        # Modes menu
        modes_menu = menubar.addMenu('Modes')

        self.continuous_draw_action = QAction('Continuous Draw Mode', self, checkable=True)
        self.continuous_draw_action.triggered.connect(self.toggle_continuous_draw_mode)
        modes_menu.addAction(self.continuous_draw_action)

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

        # Settings menu for Downscale Factor
        settings_menu = menubar.addMenu('Settings')

        set_downscale_action = QAction('Set Downscale Factor', self)
        set_downscale_action.triggered.connect(self.set_downscale_factor_dialog)
        settings_menu.addAction(set_downscale_action)

        # Initialize graphics scene and view
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)

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

        # Layout setup
        side_layout = QVBoxLayout()
        side_layout.addWidget(self.panning_mode_button)
        side_layout.addWidget(self.new_origin_button)
        side_layout.addWidget(self.z_prev_button)
        side_layout.addWidget(self.z_next_button)
        side_layout.addWidget(self.z_label)

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

    def get_adjusted_point_size(self):
        return self.annotation_point_size * self.scale_factor

    def get_adjusted_line_thickness(self):
        return self.annotation_line_thickness * self.scale_factor

    def save_annotations(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Annotations", "", "JSON Files (*.json);;All Files (*)")
        if file_name:
            data = {'duct_systems': []}

            for duct_system in self.duct_systems:
                system_data = {
                    'branch_points': {
                        name: {
                            'x': point['location'].x(),
                            'y': point['location'].y(),
                            'z': point['z']
                        }
                        for name, point in duct_system.branch_points.items()
                    },
                    'segments': {
                        name: {
                            'start_bp': segment.start_bp,
                            'end_bp': segment.end_bp,
                            'internal_points': [{'x': p[0], 'y': p[1]} for p in segment.get_internal_points()],
                            'start_z': segment.start_z,
                            'end_z': segment.end_z,
                            'annotations': [{'name': a['name'], 'x': a['x'], 'y': a['y']} for a in segment.annotations],
                            'properties': segment.properties
                        }
                        for name, segment in duct_system.segments.items()
                    }
                }
                data['duct_systems'].append(system_data)

            with open(file_name, 'w') as file:
                json.dump(data, file, indent=4)

            self.statusBar().showMessage("Annotations saved successfully.")

    def configure_segment_annotations(self):
        # Close any existing dialog
        if hasattr(self, 'annotation_config_dialog') and self.annotation_config_dialog.isVisible():
            self.annotation_config_dialog.close()

        dialog = QDialog(self)
        dialog.setWindowTitle("Configure Segment Annotations")
        dialog.resize(400, 300)

        layout = QVBoxLayout()

        # Create and clear the list widget
        self.annotations_list_widget = QListWidget()
        self.annotations_list_widget.clear()
        for annotation_name in self.segment_annotations.keys():
            item = QListWidgetItem(annotation_name)
            keybinding = self.segment_annotations[annotation_name]
            item.setData(Qt.UserRole, keybinding)
            item.setText(f"{annotation_name} (Key: {keybinding})")
            self.annotations_list_widget.addItem(item)
        layout.addWidget(self.annotations_list_widget)

        # Buttons for add, edit, and remove
        button_layout = QHBoxLayout()
        add_button = QPushButton("Add", dialog)
        add_button.clicked.connect(self.add_segment_annotation)
        button_layout.addWidget(add_button)

        edit_button = QPushButton("Edit", dialog)
        edit_button.clicked.connect(self.edit_segment_annotation)
        button_layout.addWidget(edit_button)

        remove_button = QPushButton("Remove", dialog)
        remove_button.clicked.connect(self.remove_segment_annotation)
        button_layout.addWidget(remove_button)
        layout.addLayout(button_layout)

        # Close button
        close_button = QPushButton("Close", dialog)
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        self.annotation_config_dialog = dialog
        dialog.exec_()  # Modal dialog

        # After closing the dialog, refresh the segment drawings so the new colors take effect
        self.redraw_all_segments()

    def redraw_all_segments(self):
        # Clear current segment graphics (but not the entire scene)
        for duct_system, segments in self.segment_items.items():
            for segment_name, segment_lines in segments.items():
                for line_item in segment_lines:
                    self.scene.removeItem(line_item)
        self.segment_items.clear()

        # Redraw each segment using updated annotation colors
        for duct_system in self.duct_systems:
            for segment_name, segment in duct_system.segments.items():
                self.draw_segment_with_intermediates(
                    segment.start_bp, segment.end_bp,
                    list(segment.internal_points),
                    color_key=segment_name,
                    duct_system=duct_system
                )

    def add_segment_annotation(self):
        annotation_name, ok = QInputDialog.getText(self, "Add Annotation", "Enter annotation name:")
        if ok and annotation_name:
            keybinding, ok = QInputDialog.getText(self, "Set Keybinding", "Press a key for this annotation:")
            if ok and keybinding:
                key = keybinding.upper()
                # Reserved keys to avoid conflicts
                reserved_keys = {'O', 'M', 'N', 'Z', 'X', 'P', 'D', 'Delete', 'Backspace', 'Escape'}
                if key in self.segment_annotations.values() or key in reserved_keys:
                    QMessageBox.warning(self, "Keybinding Error", f"Key '{key}' is already assigned or reserved.")
                    return

                # Now ask the user to select a color
                color = QColorDialog.getColor()
                if color.isValid():
                    self.annotation_colors[annotation_name] = color
                else:
                    # Use default color if none selected
                    self.annotation_colors[annotation_name] = Qt.red
                self.segment_annotations[annotation_name] = key
                self.save_config()
                self.configure_segment_annotations()  # Refresh the dialog
                self.load_annotations_for_current_z()
                QApplication.processEvents()  # Force GUI update
            else:
                QMessageBox.warning(self, "Input Error", "Invalid keybinding.")
        else:
            QMessageBox.warning(self, "Input Error", "Annotation name cannot be empty.")

    def edit_segment_annotation(self):
        selected_items = self.annotations_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Selection Error", "No annotation selected.")
            return
        item = selected_items[0]
        old_name = item.text().split(" (Key:")[0]
        old_key = self.segment_annotations[old_name]

        # Edit name
        new_name, ok = QInputDialog.getText(self, "Edit Annotation", "Enter new annotation name:", text=old_name)
        if ok and new_name:
            # Edit keybinding
            new_keybinding, ok = QInputDialog.getText(self, "Set Keybinding", "Press a key for this annotation:",
                                                      text=old_key)
            if ok and new_keybinding:
                new_key = new_keybinding.upper()
                reserved_keys = {'O', 'D', 'M', 'N', 'Z', 'X', 'P', 'Delete', 'Backspace', 'Escape'}
                if new_key in self.segment_annotations.values() and new_key != old_key or new_key in reserved_keys:
                    QMessageBox.warning(self, "Keybinding Error", f"Key '{new_key}' is already assigned or reserved.")
                    return
                # Ask if the user wants to change the color
                reply = QMessageBox.question(
                    self, 'Change Color?',
                    'Do you want to change the annotation color?',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    color = QColorDialog.getColor()
                    if color.isValid():
                        self.annotation_colors[new_name] = color
                    else:
                        self.annotation_colors[new_name] = self.annotation_colors.get(old_name, Qt.red)
                else:
                    # Keep the old color
                    self.annotation_colors[new_name] = self.annotation_colors.get(old_name, Qt.red)

                # Remove the old color mapping if the name changed
                if old_name != new_name and old_name in self.annotation_colors:
                    del self.annotation_colors[old_name]

                # Update the annotation mapping
                del self.segment_annotations[old_name]
                self.segment_annotations[new_name] = new_key

                # Update all segments that use the old annotation name
                for duct_system in self.duct_systems:
                    for segment in duct_system.segments.values():
                        if segment.get_property('Annotation') == old_name:
                            segment.add_property('Annotation', new_name)

                self.save_config()
                self.configure_segment_annotations()  # Refresh the dialog
                self.load_annotations_for_current_z()  # Refresh displayed annotations
                self.redraw_all_segments()  # Recolor segments with the new annotation name
                QApplication.processEvents()  # Force GUI update
            else:
                QMessageBox.warning(self, "Input Error", "Invalid keybinding.")
        else:
            QMessageBox.warning(self, "Input Error", "Annotation name cannot be empty.")

    def remove_segment_annotation(self):
        selected_items = self.annotations_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Selection Error", "No annotation selected.")
            return
        item = selected_items[0]
        annotation_name = item.text().split(" (Key:")[0]
        del self.segment_annotations[annotation_name]
        if annotation_name in self.annotation_colors:
            del self.annotation_colors[annotation_name]
        self.save_config()
        self.configure_segment_annotations()  # Refresh the dialog
        self.load_annotations_for_current_z()  # Refresh the display
        QApplication.processEvents()  # Force GUI update

    def save_config(self):
        config_data = {
            'segment_annotations': self.segment_annotations,
            'annotation_colors': {name: color.name() for name, color in self.annotation_colors.items()}
        }
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f)

    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
                self.segment_annotations = config_data.get('segment_annotations', {})
                self.annotation_colors = {}
                annotation_colors = config_data.get('annotation_colors', {})
                for name, color_name in annotation_colors.items():
                    self.annotation_colors[name] = QColor(color_name)
        except FileNotFoundError:
            # No config file exists yet
            pass

    def load_tiff(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load TIFF", "", "TIFF Files (*.tiff; *.tif);;All Files (*)",
            options=options
        )
        if file_name:
            try:
                img_array = tifffile.imread(file_name)
                # Determine the axes order using the metadata
                with tifffile.TiffFile(file_name) as tif:
                    series = tif.series[0]
                    axes = series.axes  # e.g., 'TCZYX' or 'CZYX' or 'YXS'
                    img_array = series.asarray()
                    # Parse axes to rearrange img_array dimensions
                    axes = axes.replace('S', 'C')  # Treat 'S' (Samples) as 'C' (Channels)
                    # Ensure required axes are present
                    required_axes = {'Y', 'X'}
                    if not required_axes.issubset(set(axes)):
                        QMessageBox.warning(self, "Load Image",
                                            f"Image axes {axes} do not contain required axes {required_axes}.")
                        return

                    # Initialize list of axes labels and indices
                    current_axes = list(axes)

                    # Add Z axis if missing
                    if 'Z' not in current_axes:
                        img_array = img_array[np.newaxis, ...]
                        current_axes = ['Z'] + current_axes

                    # Add C axis if missing
                    if 'C' not in current_axes:
                        img_array = img_array[..., np.newaxis]
                        current_axes = current_axes + ['C']

                    # Now, build a mapping from axes labels to indices
                    dims = {axis: i for i, axis in enumerate(current_axes)}

                    # Rearrange axes to 'Z', 'C', 'Y', 'X'
                    desired_axes = ['Z', 'C', 'Y', 'X']
                    source_indices = [dims[axis] for axis in desired_axes]
                    img_array = np.transpose(img_array, source_indices)
                    # Now img_array has shape (Z, C, Y, X)

                num_z_slices, num_channels, height, width = img_array.shape
                # Store channels separately
                self.channels = {}
                self.channel_brightness = {}
                self.downscaled_channels = {}

                for c in range(num_channels):
                    channel_name = f"Channel{c + 1}"
                    channel_data = img_array[:, c, :, :]  # Shape: (num_z_slices, height, width)
                    self.channels[channel_name] = channel_data
                    self.channel_brightness[channel_name] = 1.0

                    # Now create downscaled versions
                    downscaled_data = []
                    for z in range(num_z_slices):
                        image = channel_data[z]
                        height, width = image.shape
                        new_height = int(height * self.scale_factor)
                        new_width = int(width * self.scale_factor)
                        downscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        downscaled_data.append(downscaled_image)
                    downscaled_data = np.array(downscaled_data)
                    self.downscaled_channels[channel_name] = downscaled_data

                self.total_z_slices = num_z_slices
                self.current_z = 0
                self.display_current_z_slice()
                # Enable annotation buttons now that image is loaded

            except Exception as e:
                QMessageBox.warning(self, "Load Image", f"Failed to load image: {e}")

    def eventFilter(self, source, event):
        """Event filter to capture key presses for Z-slice navigation."""
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Up:
                self.prev_z_slice()  # Navigate to the previous Z slice
                self.statusBar().showMessage("Navigated to the previous Z slice (Up arrow).")
                self.status_message_timer.start()  # Start timer to reset status message
                return True
            elif event.key() == Qt.Key_Down:
                self.next_z_slice()  # Navigate to the next Z slice
                self.statusBar().showMessage("Navigated to the next Z slice (Down arrow).")
                self.status_message_timer.start()  # Start timer to reset status message
                return True
        return super().eventFilter(source, event)

    def clear_status_message(self):
        """Clear the status bar message."""
        self.statusBar().clearMessage()

    def display_current_z_slice(self):
        if self.downscaled_channels and self.total_z_slices > 0:
            # Combine channels with brightness settings
            combined_frame = self.combine_channels()
            if combined_frame is None:
                return  # Error message already shown in combine_channels

            # Convert combined_frame to QImage
            height, width = combined_frame.shape[:2]
            bytes_per_line = 3 * width
            qimage = QImage(
                combined_frame.data.tobytes(),
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(qimage)

            if not hasattr(self, 'pixmap_item') or not self.pixmap_item:
                # Create the pixmap item if it does not exist
                self.pixmap_item = QGraphicsPixmapItem(pixmap)
                self.pixmap_item.setTransformationMode(Qt.SmoothTransformation)
                self.scene.addItem(self.pixmap_item)
                self.scene.setSceneRect(self.pixmap_item.boundingRect())
                self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            else:
                # Update the existing pixmap without removing it
                self.pixmap_item.setPixmap(pixmap)
                self.pixmap_item.setPos(0, 0)  # Ensure consistent positioning

                # Update the scene rectangle to the pixmap's bounding rectangle
                self.scene.setSceneRect(self.pixmap_item.boundingRect())

                # Preserve the current center of the view
                current_center = self.view.mapToScene(self.view.viewport().rect().center())
                self.view.centerOn(current_center)

            # Update the Z slice label and annotations
            self.z_label.setText(f"Z Slice: {self.current_z + 1}/{self.total_z_slices}")
            self.load_annotations_for_current_z()
            self.display_outlines()  # Add this line
        else:
            QMessageBox.warning(self, "Display Image", "No channels loaded.")

    def set_downscale_factor_dialog(self):
        """Open a dialog to set the downscale factor."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Set Downscale Factor")
        dialog.resize(300, 150)

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Select Downscale Factor (0.01 - 1.0):"))

        self.downscale_spinbox = QDoubleSpinBox(dialog)
        self.downscale_spinbox.setRange(0.01, 1.0)
        self.downscale_spinbox.setSingleStep(0.05)
        self.downscale_spinbox.setValue(self.scale_factor)
        layout.addWidget(self.downscale_spinbox)

        # Apply and Close buttons
        button_layout = QHBoxLayout()
        apply_button = QPushButton("Apply", dialog)
        apply_button.clicked.connect(lambda: self.apply_downscale_factor(dialog))
        button_layout.addWidget(apply_button)

        close_button = QPushButton("Close", dialog)
        close_button.clicked.connect(dialog.accept)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        dialog.exec_()

    def apply_downscale_factor(self, dialog):
        """Apply the new downscale factor and update images."""
        new_scale = self.downscale_spinbox.value()
        if new_scale <= 0:
            QMessageBox.warning(self, "Invalid Scale Factor", "Scale factor must be greater than 0.")
            return
        self.scale_factor = new_scale
        self.statusBar().showMessage(f"Downscale factor set to {self.scale_factor}. Reprocessing images...")

        if self.channels:
            self.recompute_downscaled_channels()
            self.display_current_z_slice()
            self.statusBar().showMessage(f"Downscale factor updated to {self.scale_factor}. Images reprocessed.")
        else:
            self.statusBar().showMessage("Downscale factor updated. Load an image to apply the new scale.")

        dialog.accept()

    def recompute_downscaled_channels(self):
        """Recompute the downscaled channels based on the current scale factor."""
        self.downscaled_channels = {}
        for c in self.channels.keys():
            channel_data = self.channels[c]  # Shape: (Z, Y, X)
            downscaled_data = []
            for z in range(self.total_z_slices):
                image = channel_data[z]
                height, width = image.shape
                new_height = max(1, int(height * self.scale_factor))
                new_width = max(1, int(width * self.scale_factor))
                downscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                downscaled_data.append(downscaled_image)
            self.downscaled_channels[c] = np.array(downscaled_data)

    def load_annotations(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Annotations", "", "JSON Files (*.json);;All Files (*)")
        if file_name:
            with open(file_name, 'r') as file:
                data = json.load(file)

                # Clear existing duct systems
                self.duct_systems.clear()
                self.point_items.clear()
                self.segment_items.clear()
                self.active_duct_system = None
                self.current_point_name = None
                self.active_segment_name = None
                # Do not reset next_bp_name here

                # Check if the file contains duct_systems, else assume it's an older format
                if 'duct_systems' in data:
                    duct_systems_data = data['duct_systems']
                else:
                    # Handle older format: assume data directly holds branch_points and segments
                    duct_systems_data = [{'branch_points': data.get('branch_points', {}),
                                          'segments': data.get('segments', {})}]

                for system_data in duct_systems_data:
                    duct_system = DuctSystem()

                    # Load branch points
                    for name, point in system_data.get('branch_points', {}).items():
                        point_qt = QPointF(point['x'], point['y'])
                        z = point.get('z', 0)
                        duct_system.add_branch_point(name, point_qt, z)

                    # Load segments
                    for segment_name, segment in system_data.get('segments', {}).items():
                        start_bp = segment['start_bp']
                        end_bp = segment['end_bp']
                        intermediate_points = [(p['x'], p['y']) for p in segment['internal_points']]
                        start_z = segment.get('start_z', 0)
                        end_z = segment.get('end_z', 0)

                        duct_system.add_segment(start_bp, end_bp, segment_name, intermediate_points)
                        segment_obj = duct_system.segments[segment_name]
                        segment_obj.set_z_coordinates(start_z, end_z)

                        # Load annotations for the segment
                        for annotation in segment.get('annotations', []):
                            segment_obj.add_annotation(annotation)

                        # Load properties
                        properties = segment.get('properties', {})
                        segment_obj.properties = properties

                    self.duct_systems.append(duct_system)

                # Set the first duct system as active
                if self.duct_systems:
                    self.active_duct_system = self.duct_systems[0]
                else:
                    self.active_duct_system = DuctSystem()
                    self.duct_systems.append(self.active_duct_system)

                # After loading annotations, update next_bp_name
                max_bp_number = 0
                for duct_system in self.duct_systems:
                    for bp_name in duct_system.branch_points.keys():
                        # Extract numerical part from bp_name
                        import re
                        match = re.search(r'\d+', bp_name)
                        if match:
                            bp_number = int(match.group())
                            if bp_number > max_bp_number:
                                max_bp_number = bp_number
                self.next_bp_name = max_bp_number + 1
                # Now, when you add new branch points, they will have unique names

                self.load_annotations_for_current_z()
                self.statusBar().showMessage("Annotations loaded successfully.")

    def combine_channels(self):
        # Initialize an empty array for the combined image
        first_channel_data = next(iter(self.downscaled_channels.values()))
        height, width = first_channel_data[self.current_z].shape
        combined_frame = np.zeros((height, width, 3), dtype=np.float32)  # Use float32 for accumulation

        for idx, (channel_name, channel_data) in enumerate(self.downscaled_channels.items()):
            frame = channel_data[self.current_z].astype(np.float32)
            brightness = self.channel_brightness.get(channel_name, 1.0)
            frame_adjusted = frame * brightness

            color = np.array(self.get_channel_color(idx)) / 255.0  # Normalize color to [0,1]
            # Reshape color to (1,1,3) for broadcasting
            color = color.reshape(1, 1, 3)
            # Expand frame to (height,width,1) for broadcasting
            frame_adjusted = frame_adjusted[:, :, np.newaxis]
            # Multiply frame by color and accumulate
            combined_frame += frame_adjusted * color

        # Clip the combined frame to [0,255] and convert to uint8
        combined_frame = np.clip(combined_frame, 0, 255).astype(np.uint8)
        return combined_frame

    def get_channel_color(self, index):
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 255, 255) # White
        ]
        return colors[index % len(colors)]

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

        # Load annotations for all duct systems
        for duct_system in self.duct_systems:
            is_active = (duct_system == self.active_duct_system)
            opacity = 1.0 if is_active else 0.3  # Lower opacity for non-active systems

            # Keep track of branch points to draw
            branch_points_to_draw = {}

            # Load segments and determine which branch points to draw
            for segment_name, segment in duct_system.segments.items():
                start_bp = duct_system.get_branch_point(segment.start_bp)
                end_bp = duct_system.get_branch_point(segment.end_bp)
                if start_bp is None or end_bp is None:
                    # Skip this segment if branch points are missing
                    continue
                start_z = start_bp['z']
                end_z = end_bp['z']

                # Determine if the segment is connected to the current Z slice
                if start_z == self.current_z or end_z == self.current_z:
                    # Add branch points to draw
                    branch_points_to_draw[segment.start_bp] = start_bp
                    branch_points_to_draw[segment.end_bp] = end_bp

            if is_active and self.current_point_name:
                active_bp = duct_system.get_branch_point(self.current_point_name)
                if active_bp and self.current_point_name not in branch_points_to_draw:
                    # Compute the color using the same delta_z logic
                    display_point = QPointF(active_bp["location"].x() * self.scale_factor,
                                            active_bp["location"].y() * self.scale_factor)
                    delta_z = active_bp['z'] - self.current_z
                    max_delta_z = 5
                    delta_z_capped = max(-max_delta_z, min(delta_z, max_delta_z))
                    hue_start = 0
                    hue_middle = 120
                    hue_end = 240
                    if delta_z_capped == 0:
                        hue = hue_middle
                    elif delta_z_capped > 0:
                        hue = hue_middle + (delta_z_capped / max_delta_z) * (hue_end - hue_middle)
                    else:
                        hue = hue_middle + (delta_z_capped / max_delta_z) * (hue_middle - hue_start)
                    hue = int(hue) % 360
                    base_color = QColor.fromHsv(hue, 255, 255)
                    adjusted_point_size = self.get_adjusted_point_size()
                    point_item = self.scene.addEllipse(
                        display_point.x() - (adjusted_point_size / 2),
                        display_point.y() - (adjusted_point_size / 2),
                        adjusted_point_size, adjusted_point_size,
                        QPen(base_color), QBrush(base_color)
                    )
                    self.point_items.setdefault(duct_system, {})[self.current_point_name] = point_item

            # Draw branch points
            for name, point in branch_points_to_draw.items():
                point_qt = QPointF(point['location'].x(), point['location'].y())
                # Adjust point_qt to displayed coordinates
                display_point = QPointF(point_qt.x() * self.scale_factor, point_qt.y() * self.scale_factor)
                delta_z = point['z'] - self.current_z
                max_delta_z = 5  # Adjust as needed

                # Cap delta_z to be within [-max_delta_z, max_delta_z]
                delta_z_capped = max(-max_delta_z, min(delta_z, max_delta_z))

                # Map delta_z_capped to hue value
                hue_start = 0    # Red
                hue_middle = 120 # Green
                hue_end = 240    # Blue

                if delta_z_capped == 0:
                    hue = hue_middle
                elif delta_z_capped > 0:
                    # Higher slices: Green to Blue
                    hue = hue_middle + (delta_z_capped / max_delta_z) * (hue_end - hue_middle)
                else:
                    # Lower slices: Red to Green
                    hue = hue_middle + (delta_z_capped / max_delta_z) * (hue_middle - hue_start)

                hue = int(hue) % 360  # Ensure hue is within 0-359 degrees

                color = QColor.fromHsv(hue, 255, 255)  # Full saturation and value

                adjusted_point_size = self.get_adjusted_point_size()
                point_item = self.scene.addEllipse(
                    display_point.x() - (adjusted_point_size / 2), display_point.y() - (adjusted_point_size / 2),
                    adjusted_point_size, adjusted_point_size,
                    QPen(color), QBrush(color)
                )
                point_item.setOpacity(opacity)
                self.point_items.setdefault(duct_system, {})[name] = point_item

            # Draw segments connected to the current Z slice
            for segment_name, segment in duct_system.segments.items():
                start_bp = duct_system.get_branch_point(segment.start_bp)
                end_bp = duct_system.get_branch_point(segment.end_bp)
                start_z = start_bp['z']
                end_z = end_bp['z']

                if start_z == self.current_z or end_z == self.current_z:
                    # Determine if the segment connects to a different Z slice
                    different_style = start_z != end_z
                    self.draw_segment_with_intermediates(
                        segment.start_bp, segment.end_bp,
                        list(segment.internal_points),
                        color_key=segment_name,
                        opacity=opacity,
                        duct_system=duct_system,
                        different_style=different_style
                    )

                    # Load annotations on the current Z slice
                    for annotation in segment.annotations:
                        if annotation.get('z', self.current_z) == self.current_z:
                            point = QPointF(annotation['x'], annotation['y'])
                            # Adjust point to displayed coordinates
                            display_point = QPointF(point.x() * self.scale_factor, point.y() * self.scale_factor)
                            if 'color' in annotation:
                                annotation_color = QColor(annotation['color'])
                            else:
                                annotation_color = self.annotation_colors.get(annotation['name'], Qt.red)

                            adjusted_point_size = self.get_adjusted_point_size()
                            annotation_item = self.scene.addEllipse(
                                display_point.x() - (adjusted_point_size / 2), display_point.y() - (adjusted_point_size / 2),
                                adjusted_point_size, adjusted_point_size,
                                QPen(annotation_color), QBrush(annotation_color)
                            )
                            annotation_item.setOpacity(opacity)

    def clear_annotations(self):
        # Remove all items except the image and the outline items
        for item in self.scene.items():
            if isinstance(item, QGraphicsPixmapItem):
                continue
            if item in self.outline_items:
                continue
            self.scene.removeItem(item)
        self.point_items.clear()
        self.segment_items.clear()

    def handle_mouse_press(self, event):
        if self.panning_mode:
            self.pan_start = event.pos()  # Capture the mouse position at the start of the panning
            self.view.setCursor(Qt.ClosedHandCursor)
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
                    if self.new_origin_mode:
                        self.set_origin(point)
                        self.new_origin_mode = False
                        self.statusBar().showMessage("New origin set.")
                        self.view.setCursor(QCursor(Qt.ArrowCursor))
                        self.set_active_point(self.current_point_name)
                    else:
                        self.handle_left_click(point)
                elif event.button() == Qt.RightButton:
                    if self.continuous_draw_mode:
                        self.drawing_continuous = True
                        self.add_intermediate_point(point)
                    elif self.current_point_name is not None:
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
        elif self.new_origin_mode:
            self.clear_temp_line()
        elif not self.selection_mode and not self.panning_mode and not self.new_origin_mode and self.current_point_name is not None:
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

    def add_branch_point(self, point, bp_name=None, z=None):
        if bp_name is None:
            bp_name = f"bp{self.next_bp_name}"
            self.next_bp_name += 1

        if z is None:
            z = self.current_z

        # Map point to original image coordinates
        original_point = QPointF(point.x() / self.scale_factor, point.y() / self.scale_factor)

        self.active_duct_system.add_branch_point(bp_name, original_point, z)
        color = Qt.green

        adjusted_point_size = self.get_adjusted_point_size()
        point_item = self.scene.addEllipse(
            point.x() - (adjusted_point_size / 2), point.y() - (adjusted_point_size / 2),
            adjusted_point_size, adjusted_point_size,
            QPen(color), QBrush(color)
        )

        # Ensure that the dictionary for the active duct system exists
        if self.active_duct_system not in self.point_items:
            self.point_items[self.active_duct_system] = {}
        self.point_items[self.active_duct_system][bp_name] = point_item

        self.set_active_point(bp_name)
        self.statusBar().showMessage(f"Branch point '{bp_name}' created at {point} on Z slice {z}.")

    def ensure_qpointf(self, point):
        if isinstance(point, tuple):
            return QPointF(point[0], point[1])
        return point  # If it's already a QPointF, return it as is

    def add_intermediate_point(self, point, color=Qt.gray):
        # Convert the last point to QPointF and ensure it's a copy
        if self.intermediate_points:
            last_point_original = self.ensure_qpointf(self.intermediate_points[-1])
            last_point_display = QPointF(last_point_original.x() * self.scale_factor,
                                         last_point_original.y() * self.scale_factor)
        else:
            last_point_original = self.active_duct_system.get_branch_point(self.current_point_name)["location"]
            last_point_display = QPointF(last_point_original.x() * self.scale_factor,
                                         last_point_original.y() * self.scale_factor)

        # Ensure the current point is a QPointF and store a copy as tuple
        display_point = self.ensure_qpointf(point)
        original_point = QPointF(display_point.x() / self.scale_factor, display_point.y() / self.scale_factor)

        adjusted_line_thickness = 2 * self.scale_factor
        # Draw line between last_point_display and display_point
        line = self.scene.addLine(
            last_point_display.x(), last_point_display.y(),
            display_point.x(), display_point.y(),
            QPen(color, adjusted_line_thickness)
        )
        self.dotted_lines.append(line)

        # Store the point as a tuple (x, y) in original image coordinates
        self.intermediate_points.append((original_point.x(), original_point.y()))

        self.statusBar().showMessage(f"Intermediate point added at {original_point}.")

    def finalize_segment(self, end_point):
        bp_name = f"bp{self.next_bp_name}"
        z = self.current_z
        # Map end_point to original coordinates
        original_end_point = QPointF(end_point.x() / self.scale_factor, end_point.y() / self.scale_factor)
        self.active_duct_system.add_branch_point(bp_name, original_end_point, z)

        segment_name = f"{self.current_point_name}to{bp_name}"
        self.active_duct_system.add_segment(
            self.current_point_name, bp_name, segment_name,
            list(self.intermediate_points)
        )
        self.active_duct_system.segments[segment_name].set_z_coordinates(
            self.active_duct_system.get_branch_point(self.current_point_name)["z"], z
        )

        # Assign active annotation
        if self.active_segment_annotation:
            self.active_duct_system.segments[segment_name].add_property('Annotation', self.active_segment_annotation)
            self.statusBar().showMessage(
                f"Segment '{segment_name}' created with annotation '{self.active_segment_annotation}'.")
        else:
            self.statusBar().showMessage(f"Segment '{segment_name}' created on Z slice {z}.")

        # Draw the segment
        self.draw_segment_with_intermediates(
            self.current_point_name, bp_name,
            list(self.intermediate_points),
            color_key=segment_name,
            duct_system=self.active_duct_system  # Pass the active duct system
        )

        color = Qt.green

        adjusted_point_size = self.get_adjusted_point_size()
        point_item = self.scene.addEllipse(
            end_point.x() - (adjusted_point_size / 2), end_point.y() - (adjusted_point_size / 2),
            adjusted_point_size, adjusted_point_size,
            QPen(color), QBrush(color)
        )
        point_item.setOpacity(1.0)
        self.point_items.setdefault(self.active_duct_system, {})[bp_name] = point_item

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

        start_point_original = self.active_duct_system.get_branch_point(self.current_point_name)["location"]
        start_point_display = QPointF(start_point_original.x() * self.scale_factor, start_point_original.y() * self.scale_factor)

        # Draw temporary line from the last point (or start) to the current mouse position
        if self.intermediate_points:
            last_intermediate = self.intermediate_points[-1]
            last_point_original = QPointF(last_intermediate[0], last_intermediate[1])
            last_point_display = QPointF(last_point_original.x() * self.scale_factor,
                                         last_point_original.y() * self.scale_factor)
        else:
            last_point_display = start_point_display

        adjusted_line_thickness = 2 * self.scale_factor
        self.temp_line = self.scene.addLine(
            last_point_display.x(), last_point_display.y(),
            point.x(), point.y(),
            QPen(Qt.gray, adjusted_line_thickness, Qt.DashLine)
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
        for duct_system in self.duct_systems:
            for bp_name, bp in duct_system.branch_points.items():
                bp_location = bp["location"]
                # Map bp_location to displayed coordinates
                bp_display_location = QPointF(bp_location.x() * self.scale_factor, bp_location.y() * self.scale_factor)
                if self.is_point_near(bp_display_location, point):
                    self.active_duct_system = duct_system
                    self.load_annotations_for_current_z()
                    self.set_active_point(bp_name)
                    return

    def set_active_point(self, bp_name):
        if self.current_point_name and self.current_point_name in self.point_items.get(self.active_duct_system, {}):
            old_active_item = self.point_items[self.active_duct_system][self.current_point_name]
            bp = self.active_duct_system.get_branch_point(self.current_point_name)
            color = Qt.green
            old_active_item.setBrush(QBrush(color))

        if bp_name in self.point_items.get(self.active_duct_system, {}):
            self.current_point_name = bp_name
            active_item = self.point_items[self.active_duct_system][bp_name]
            active_item.setBrush(QBrush(Qt.magenta))  # Mark active point
            self.statusBar().showMessage(f"Active point set to '{bp_name}'.")
        else:
            self.current_point_name = None  # Clear the current point if it doesn't exist
            self.statusBar().showMessage("No active point set.")

    def draw_segment_with_intermediates(self, start_bp_name, end_bp_name, intermediate_points, color_key=None,
                                        opacity=1.0, duct_system=None, different_style=False):
        if duct_system is None:
            duct_system = self.active_duct_system

        start_bp = duct_system.get_branch_point(start_bp_name)
        end_bp = duct_system.get_branch_point(end_bp_name)
        start_point = start_bp["location"]
        end_point = end_bp["location"]

        # Build the full list of points
        full_points = [start_point] + [QPointF(x, y) for x, y in intermediate_points] + [end_point]

        segment_lines = []  # List to store all lines for the segment

        # Use the default line color
        color = self.default_line_color if hasattr(self, 'default_line_color') else Qt.blue

        # Adjust color based on segment annotation
        segment = duct_system.get_segment(f"{start_bp_name}to{end_bp_name}")
        if segment:
            annotation = segment.get_property('Annotation')
            if annotation:
                # Use the color you selected for this annotation (fallback to default if not set)
                color = self.annotation_colors.get(annotation, self.default_line_color if hasattr(self,
                                                                                                  'default_line_color') else Qt.blue)

        # If different_style is True, adjust pen style or color
        adjusted_line_thickness = self.get_adjusted_line_thickness()
        if different_style:
            pen = QPen(Qt.gray, adjusted_line_thickness, Qt.DashLine)
        else:
            pen = QPen(color, adjusted_line_thickness)

        # Draw lines between consecutive points
        for i in range(len(full_points) - 1):
            p1 = full_points[i]
            p2 = full_points[i + 1]

            # Map points to display coordinates
            p1_display = QPointF(p1.x() * self.scale_factor, p1.y() * self.scale_factor)
            p2_display = QPointF(p2.x() * self.scale_factor, p2.y() * self.scale_factor)

            line = self.scene.addLine(
                p1_display.x(), p1_display.y(),
                p2_display.x(), p2_display.y(),
                pen
            )
            line.setOpacity(opacity)
            segment_lines.append(line)

        # Store the list of line items for the segment
        segment_name = f"{start_bp_name}to{end_bp_name}"
        self.segment_items.setdefault(duct_system, {})[segment_name] = segment_lines

    def is_point_near(self, bp_location, click_point, threshold=10):
        return (bp_location - click_point).manhattanLength() < threshold

    def load_outlines(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load Outlines", "", "JSON Files (*.geojson);;All Files (*)",
            options=options
        )
        if file_name:
            # Ask the user to select a color for the outlines
            color = QColorDialog.getColor()
            if color.isValid():
                self.outline_color = color
            else:
                self.outline_color = Qt.red  # Default to red if no color selected

            # Now open and parse the JSON file
            try:
                with open(file_name, 'r') as f:
                    data = json.load(f)
                    # Check that data has the expected structure
                    if 'features' in data:
                        features = data['features']
                        # Clear existing outlines data
                        self.outlines_data.clear()
                        # Iterate over features
                        for feature in features:
                            geometry = feature.get('geometry')
                            if geometry:
                                geom_type = geometry.get('type')
                                coordinates = geometry.get('coordinates', [])
                                if geom_type == 'Polygon':
                                    # Iterate over all rings in the Polygon
                                    for ring in coordinates:
                                        points = ring
                                        self.outlines_data.append({'points': points})
                                elif geom_type == 'MultiPolygon':
                                    # Iterate over all Polygons and their rings
                                    for polygon in coordinates:
                                        for ring in polygon:
                                            points = ring
                                            self.outlines_data.append({'points': points})
                        # Now display the outlines
                        self.display_outlines()
                    else:
                        QMessageBox.warning(self, "Load Outlines", "Invalid JSON format: 'features' key not found.")
            except Exception as e:
                QMessageBox.warning(self, "Load Outlines", f"Failed to load outlines: {e}")

    def display_outlines(self):
        # Remove existing outline items from the scene
        for item in self.outline_items:
            self.scene.removeItem(item)
        self.outline_items.clear()

        # Iterate over the outlines data
        for outline in self.outlines_data:
            points = outline['points']
            # Scale the points
            polygon = QPolygonF([QPointF(x * self.scale_factor, y * self.scale_factor) for x, y in points])
            # Create QGraphicsPolygonItem
            polygon_item = QGraphicsPolygonItem(polygon)
            # Set pen color
            pen = QPen(self.outline_color, 2)
            polygon_item.setPen(pen)
            # Add to scene
            self.scene.addItem(polygon_item)
            # Store the item
            self.outline_items.append(polygon_item)

    def keyPressEvent(self, event):
        key = event.text().upper()
        if key in self.segment_annotations.values():
            # Toggle the annotation
            annotation_name = [name for name, k in self.segment_annotations.items() if k == key][0]
            if self.active_segment_annotation == annotation_name:
                self.active_segment_annotation = None
                self.statusBar().showMessage(f"Annotation '{annotation_name}' deactivated.")
            else:
                self.active_segment_annotation = annotation_name
                self.statusBar().showMessage(f"Annotation '{annotation_name}' activated.")
        elif event.key() == Qt.Key_M:
            self.toggle_panning_mode()
        elif event.key() == Qt.Key_O:
            self.clear_temp_line()
            self.clear_dotted_lines()
            self.clear_intermediate_points()
            self.selection_mode = True
            self.panning_mode = False
            self.update_mode_display()
            self.statusBar().showMessage("Selection mode: Click on a branch point to select it as the active point.")
        elif event.key() == Qt.Key_Escape:
            if self.panning_mode:
                self.toggle_panning_mode()
            else:
                self.reset_modes()
        elif event.key() == Qt.Key_Backspace:
            self.revert_to_previous_branch_point()

        elif event.key() == Qt.Key_Delete or event.key() == Qt.Key_D:
            self.delete_most_recent_branch()
        elif event.key() == Qt.Key_N:
            self.clear_temp_line()
            self.clear_dotted_lines()
            self.clear_intermediate_points()
            self.activate_new_origin_mode()
        elif event.key() == Qt.Key_Z:
            self.prev_z_slice()
            self.statusBar().showMessage("Navigated to the previous Z slice (Z key).")
            self.status_message_timer.start()
        elif event.key() == Qt.Key_X:
            self.next_z_slice()
            self.statusBar().showMessage("Navigated to the next Z slice (X key).")
            self.status_message_timer.start()
        else:
            super().keyPressEvent(event)

    def reset_modes(self):
        self.selection_mode = False
        self.annotation_mode = None
        self.custom_annotation_name = None
        self.view.setCursor(QCursor(Qt.ArrowCursor))
        self.statusBar().showMessage("Modes reset.")
        self.update_mode_display()

    def delete_branch_point_and_descendants(self, bp_name, duct_system, visited_bps=None):
        if visited_bps is None:
            visited_bps = set()
        if bp_name in visited_bps:
            return
        visited_bps.add(bp_name)

        # First, get the branch point from the duct system
        bp = duct_system.get_branch_point(bp_name)
        if bp is None:
            return  # Branch point does not exist

        # Find all segments connected to this branch point
        segments_to_delete = []
        for segment_name, segment in list(duct_system.segments.items()):
            if segment.start_bp == bp_name or segment.end_bp == bp_name:
                segments_to_delete.append(segment_name)

        # For each segment, handle deletion
        for segment_name in segments_to_delete:
            segment = duct_system.get_segment(segment_name)
            if segment.start_bp == bp_name:
                end_bp_name = segment.end_bp
                # Recursively delete downstream branch points
                self.delete_branch_point_and_descendants(end_bp_name, duct_system, visited_bps)
            # Remove the segment
            self.remove_segment(segment_name, duct_system)

        # Remove the branch point
        self.remove_branch_point_and_item(bp_name, duct_system)

    def delete_most_recent_branch(self):
        if not self.active_duct_system.branch_points:
            self.statusBar().showMessage("No branch points to delete in the active duct system.")
            return

        # Get the list of branch point names in the active duct system
        bp_names = list(self.active_duct_system.branch_points.keys())

        # Extract numerical parts from branch point names
        def get_bp_number(name):
            import re
            m = re.search(r'\d+', name)
            if m:
                return int(m.group())
            else:
                return 0

        # Sort them based on the numerical part of the branch point name
        bp_names_sorted = sorted(bp_names, key=get_bp_number)

        # Identify the most recent branch point
        last_bp_name = bp_names_sorted[-1]

        # Delete the branch point and its descendants
        self.delete_branch_point_and_descendants(last_bp_name, self.active_duct_system)

        # Update current point name
        remaining_bp_names = list(self.active_duct_system.branch_points.keys())
        if remaining_bp_names:
            remaining_bp_names_sorted = sorted(remaining_bp_names, key=get_bp_number)
            new_last_bp_name = remaining_bp_names_sorted[-1]
            self.set_active_point(new_last_bp_name)
        else:
            self.current_point_name = None
            self.statusBar().showMessage("All branch points have been deleted.")

        self.load_annotations_for_current_z()  # Refresh the display

        self.statusBar().showMessage(f"Branch point '{last_bp_name}' and downstream elements deleted.")

    def delete_selected_branch_point_and_descendants(self):
        if self.current_point_name:
            bp_name = self.current_point_name
            self.delete_branch_point_and_descendants(bp_name, self.active_duct_system)
            self.current_point_name = None
            self.load_annotations_for_current_z()
            self.statusBar().showMessage(f"Branch point '{bp_name}' and its descendants have been deleted.")
        else:
            QMessageBox.warning(self, "Delete Branch Point", "No branch point is currently selected.")

    def remove_segment(self, segment_name, duct_system):
        if segment_name in self.segment_items.get(duct_system, {}):
            # Remove the visual segment lines
            for segment_item in self.segment_items[duct_system][segment_name]:
                self.scene.removeItem(segment_item)
            del self.segment_items[duct_system][segment_name]

        if segment_name in duct_system.segments:
            segment = duct_system.segments[segment_name]
            # Remove annotations associated with this segment
            for annotation in segment.annotations:
                annotation_point = QPointF(annotation['x'], annotation['y'])
                items_at_point = self.scene.items(annotation_point)
                for item in items_at_point:
                    if isinstance(item, QGraphicsEllipseItem):
                        self.scene.removeItem(item)
            # Delete the segment from the duct system
            del duct_system.segments[segment_name]

    def remove_branch_point_and_item(self, bp_name, duct_system):
        """Helper method to remove a branch point and its associated graphic item."""
        duct_system.remove_branch_point(bp_name)
        if bp_name in self.point_items.get(duct_system, {}):
            self.scene.removeItem(self.point_items[duct_system][bp_name])
            del self.point_items[duct_system][bp_name]

    def toggle_panning_mode(self):
        self.panning_mode = not self.panning_mode
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
        mode = "Panning Mode" if self.panning_mode else "Selection Mode" if self.selection_mode else "Point Mode"
        self.panning_mode_button.setChecked(self.panning_mode)
        self.z_prev_button.setEnabled(hasattr(self, 'channels') and self.total_z_slices > 0)
        self.z_next_button.setEnabled(hasattr(self, 'channels') and self.total_z_slices > 0)
        self.new_origin_button.setEnabled(True)  # Always enabled
        self.z_label.setEnabled(hasattr(self, 'channels') and self.total_z_slices > 0)
        self.statusBar().showMessage(f"Current Mode: {mode}")

    def add_annotation_point(self, point):
        if self.annotation_mode and self.active_segment_name:
            # Map point to original image coordinates
            original_point = QPointF(point.x() / self.scale_factor, point.y() / self.scale_factor)
            # Look up the color now and store it as a string (so it’s JSON-serializable)
            chosen_color = self.annotation_colors.get(self.annotation_mode, Qt.red)
            annotation = {
                'name': self.annotation_mode,
                'x': original_point.x(),
                'y': original_point.y(),
                'z': self.current_z,
                'color': chosen_color.name()  # store the hex color string
            }
            segment = self.active_duct_system.get_segment(self.active_segment_name)
            if segment:
                segment.add_annotation(annotation)
                pen = QPen(chosen_color, self.get_adjusted_line_thickness())
                brush = QBrush(chosen_color)
                adjusted_point_size = self.get_adjusted_point_size()
                annotation_item = self.scene.addEllipse(
                    point.x() - (adjusted_point_size / 2),
                    point.y() - (adjusted_point_size / 2),
                    adjusted_point_size, adjusted_point_size,
                    pen, brush
                )
                annotation_item.setOpacity(1.0)
                self.statusBar().showMessage(
                    f"Annotation '{self.annotation_mode}' added to segment '{segment.segment_name}'."
                )

    def revert_to_previous_branch_point(self):
        if self.current_point_name:
            segments = list(self.active_duct_system.segments.values())
            for segment in reversed(segments):
                if segment.end_bp == self.current_point_name:
                    self.set_active_point(segment.start_bp)
                    self.statusBar().showMessage(f"Reverted to previous branch point '{segment.start_bp}'.")
                    # Clear the intermediate points and temp line
                    self.clear_intermediate_points()
                    self.clear_temp_line()
                    # Capture current mouse position in scene coordinates
                    mouse_pos = self.view.mapToScene(self.view.mapFromGlobal(QCursor.pos()))
                    # Redraw the temp line using the new active branch point as the start
                    self.update_temp_line(mouse_pos)
                    break
            else:
                self.statusBar().showMessage("No previous branch point to revert to.")

    def change_outline_color(self):
        if self.outlines_data:
            color = QColorDialog.getColor()
            if color.isValid():
                self.outline_color = color
                # Update the outline items
                for item in self.outline_items:
                    pen = item.pen()
                    pen.setColor(self.outline_color)
                    item.setPen(pen)
                self.statusBar().showMessage("Outline color updated.")
        else:
            QMessageBox.warning(self, "Change Outline Color", "No outlines loaded.")

    def edit_line_colors(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.default_line_color = color
            self.statusBar().showMessage("Default line color updated. Redrawing segments...")
            self.redraw_all_segments()

    def toggle_continuous_draw_mode(self):
        self.continuous_draw_mode = not self.continuous_draw_mode
        self.statusBar().showMessage("Continuous Draw Mode "
                                     f"{'Activated' if self.continuous_draw_mode else 'Deactivated'}.")

    def activate_new_origin_mode(self):
        self.new_origin_mode = True
        self.statusBar().showMessage("New Origin Mode: Click to set a new origin.")
        self.view.setCursor(QCursor(Qt.CrossCursor))

    def set_origin(self, point):
        # Create a new duct system
        self.active_duct_system = DuctSystem()
        self.duct_systems.append(self.active_duct_system)

        self.current_point_name = None  # Reset current point
        self.active_segment_name = None  # Reset active segment
        self.intermediate_points.clear()
        self.clear_temp_line()
        self.clear_dotted_lines()
        # Now add the new origin
        origin_name = f"bp{self.next_bp_name}"
        self.add_branch_point(point, origin_name)
        self.next_bp_name += 1
        self.current_point_name = origin_name  # Set the current point to the new origin
        self.statusBar().showMessage(f"New origin set at {point}.")

        # Ensure the new origin point is visible immediately
        self.load_annotations_for_current_z()

    def show_brightness_dialog(self):
        if not self.channels:
            QMessageBox.warning(self, "Adjust Brightness", "No channels loaded.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Adjust Channel Brightness")
        dialog.resize(300, 400)

        layout = QVBoxLayout()

        self.brightness_sliders = {}
        for channel in self.channels.keys():
            layout.addWidget(QLabel(f"{channel} Brightness:"))
            slider = QSlider(Qt.Horizontal, dialog)
            slider.setMinimum(1)
            slider.setMaximum(200)  # Brightness factor from 0.01 to 2.0
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
            "- Click 'Load Image(s)' to load a TIFF image with multiple channels or Z-slices.\n"
            "- Use 'Panning Mode' to move the view; scrolling always works for zooming.\n"
            "- Press 'M' to toggle Panning Mode (move around the image).\n"
            "- Press 'O' to select and activate branch points.\n"
            "- Press 'D' or 'Delete' to remove the last created branch.\n"
            "- Press 'Backspace' to revert the active branch point to the previous point.\n"
            "- Use 'Z' or 'X' to navigate between Z slices (up and down).\n"
            "- Press 'N' to create a new origin.\n"
            "- Use the 'Edit' menu to configure segment annotations and keybindings. Note that editing a name retroactively changes the previous annotations with that name as well.\n"
            "- The system uses a config.json file to store annotation colors and keybindings, it is automatically generated and updated.\n"
            "- Press the assigned toggle key to activate/deactivate an annotation. Note that activating an annotation deactivates other annotations\n"
            "- Use the 'Brightness' option in the 'Channels' menu to adjust the brightness of individual channels.\n\n"
            
            "If you have big images, I recommend setting the downscale factor down before loading it\n\n"
            
            "Developed by Jeroen Doornbos based on a version from Jacco van Rheenen."
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


# Supporting classes for Duct System
class DuctSystem:
    def __init__(self):
        self.branch_points = {}
        self.segments = {}

    def add_branch_point(self, name, location, z):
        self.branch_points[name] = {
            "name": name,
            "location": location,
            "z": z,
        }

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

    def get_segment(self, segment_name):
        return self.segments.get(segment_name)


class DuctSegment:
    def __init__(self, start_bp, end_bp, segment_name):
        self.start_bp = start_bp
        self.end_bp = end_bp
        self.segment_name = segment_name
        self.internal_points = []  # List to store internal points between branch points
        self.annotations = []  # List to store annotations
        self.start_z = None  # Z-coordinate of start branch point
        self.end_z = None  # Z-coordinate of end branch point
        self.properties = {}  # Dictionary to store segment properties

    def set_z_coordinates(self, start_z, end_z):
        self.start_z = start_z
        self.end_z = end_z

    def get_internal_points(self):
        return self.internal_points

    def add_annotation(self, annotation):
        """Add an annotation to the segment."""
        self.annotations.append(annotation)

    def add_property(self, key, value):
        self.properties[key] = value

    def get_property(self, key):
        return self.properties.get(key)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DuctSystemGUI()
    ex.show()
    sys.exit(app.exec_())
