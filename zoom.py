from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QMouseEvent, QWheelEvent

class ZoomableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._original_pixmap = None  # Store the original QPixmap
        self._current_pixmap = None  # Store the current QPixmap (zoomed or not)
        self._zoom_factor = 1.0  # Current zoom factor (1.0 = no zoom)
        self._max_zoom = 5.0  # Maximum zoom factor
        self._min_zoom = 1.0  # Minimum zoom factor
        self._is_dragging = False  # Track if the user is dragging the image
        self._drag_start_pos = QPoint()  # Track the start position of dragging
        self._drag_offset = QPoint()  # Track the offset during dragging

    def setPixmap(self, pixmap):
        """Override setPixmap to store the original and current pixmap."""
        self._original_pixmap = pixmap
        self._current_pixmap = pixmap
        self._zoom_factor = 1.0  # Reset zoom factor
        self._drag_offset = QPoint()  # Reset drag offset
        super().setPixmap(pixmap)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel events for zooming in and out."""
        if self._original_pixmap is None:
            return

        # Calculate the new zoom factor based on the wheel delta
        delta = event.angleDelta().y()
        if delta > 0:
            new_zoom_factor = min(self._zoom_factor * 1.1, self._max_zoom)  # Zoom in
        else:
            new_zoom_factor = max(self._zoom_factor / 1.1, self._min_zoom)  # Zoom out

        # Calculate the cursor position relative to the image
        cursor_pos = event.pos()
        image_center = QPoint(self.width() / 2, self.height() / 2)
        cursor_offset = cursor_pos - image_center

        # Adjust the drag offset based on the zoom factor change
        self._drag_offset = (self._drag_offset + cursor_offset) * (new_zoom_factor / self._zoom_factor) - cursor_offset

        # Update the zoom factor
        self._zoom_factor = new_zoom_factor

        # Update the displayed pixmap
        self._update_display()

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events for starting drag."""
        if event.button() == Qt.LeftButton and self._zoom_factor > self._min_zoom:
            self._is_dragging = True
            self._drag_start_pos = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events for dragging."""
        if self._is_dragging:
            # Calculate the drag offset
            delta = event.pos() - self._drag_start_pos
            self._drag_offset += delta
            self._drag_start_pos = event.pos()
            self._update_display()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events for stopping drag."""
        if event.button() == Qt.LeftButton:
            self._is_dragging = False

    def _update_display(self):
        """Update the displayed pixmap based on zoom and drag offset."""
        if self._original_pixmap is None:
            return

        # Calculate the scaled pixmap size
        scaled_size = self._original_pixmap.size() * self._zoom_factor
        self._current_pixmap = self._original_pixmap.scaled(
            scaled_size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
        )

        # Calculate the visible area of the pixmap
        visible_rect = self.rect()
        visible_rect.moveCenter(self._current_pixmap.rect().center() + self._drag_offset)

        # Crop the pixmap to the visible area
        cropped_pixmap = self._current_pixmap.copy(visible_rect)

        # Scale the cropped pixmap to fit the QLabel
        scaled_pixmap = cropped_pixmap.scaled(
            self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
        )

        # Set the scaled pixmap to the QLabel
        super().setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """Handle resize events to update the display."""
        self._update_display()

    def update_pixmap(self, pixmap):
        """Update the underlying pixmap without resetting zoom or drag."""
        self._original_pixmap = pixmap
        self._current_pixmap = pixmap
        self._update_display()  # Refresh the display with the new pixmap

    def setPixmap(self, pixmap):
        """Override setPixmap to store the original and current pixmap."""
        self.update_pixmap(pixmap)  # Use the new method to avoid duplication