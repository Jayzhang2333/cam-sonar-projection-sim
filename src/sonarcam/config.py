from pydantic import BaseModel, Field

class CameraIntrinsics(BaseModel):
    fx: float = Field(600.0, description="Focal length in pixels (x)")
    fy: float = Field(600.0, description="Focal length in pixels (y)")
    cx: float = Field(320.0, description="Principal point x")
    cy: float = Field(240.0, description="Principal point y")
    width: int = 640
    height: int = 480

class SonarIntrinsics(BaseModel):
    range_min: float = 0.5
    range_max: float = 30.0
    num_bins: int = 512
    num_beams: int = 720
    elevation_fov_deg: float = 20.0

class Extrinsics(BaseModel):
    R_cam_sonar: list[float] = Field(default_factory=lambda: [1,0,0, 0,1,0, 0,0,1])
    t_cam_sonar: list[float] = Field(default_factory=lambda: [0.2, 0.0, 0.0])

class WorldPose(BaseModel):
    position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.5])
    rpy: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])

class AppConfig(BaseModel):
    camera: CameraIntrinsics = CameraIntrinsics()
    sonar: SonarIntrinsics = SonarIntrinsics()
    extrinsics: Extrinsics = Extrinsics()
    pose: WorldPose = WorldPose()
