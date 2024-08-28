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
        self.annotations = {}  # Dictionary to store any annotations

    def add_internal_point(self, point):
        """Add an internal point to the segment."""
        self.internal_points.append(point)

    def add_annotation(self, key, value):
        """Add an annotation to the segment."""
        self.annotations[key] = value

    def get_internal_points(self):
        """Return the list of internal points."""
        return self.internal_points

    def get_annotations(self):
        """Return the annotations dictionary."""
        return self.annotations


class DuctSystem:
    def __init__(self):
        self.branch_points = []  # List of branch points
        self.segments = {}  # Dictionary of segments

    def add_branch_point(self, name, location):
        self.branch_points.append({"name": name, "location": location})

    def get_branch_point(self, name):
        return next((bp for bp in self.branch_points if bp["name"] == name), None)

    def add_segment(self, start_bp, end_bp, segment_name):
        if self.get_branch_point(start_bp) and self.get_branch_point(end_bp):
            segment = DuctSegment(start_bp, end_bp, segment_name)
            self.segments[segment_name] = segment
        else:
            print("One or both branch points do not exist.")

    def get_segment(self, segment_name):
        return self.segments.get(segment_name)

    def add_point_to_segment(self, segment_name, point):
        segment = self.get_segment(segment_name)
        if segment:
            segment.add_internal_point(point)

    def add_annotation_to_segment(self, segment_name, key, value):
        segment = self.get_segment(segment_name)
        if segment:
            segment.add_annotation(key, value)

