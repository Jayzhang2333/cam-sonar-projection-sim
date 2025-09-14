import enum
class ScanOrientation(str, enum.Enum):
    horizontal = "horizontal"
    vertical = "vertical"  # placeholder

class SonarModel:
    def __init__(self, range_min, range_max, num_bins, num_beams, elevation_fov_deg,
                 orientation: ScanOrientation, sector_start_deg, sector_end_deg):
        self.range_min=float(range_min); self.range_max=float(range_max)
        self.num_bins=int(num_bins); self.num_beams=int(num_beams)
        self.elevation_fov_deg=float(elevation_fov_deg)
        self.orientation=ScanOrientation(orientation)
        self.sector_start_deg=float(sector_start_deg)
        self.sector_end_deg=float(sector_end_deg)
