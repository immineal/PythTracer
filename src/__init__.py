import numpy as np
import multiprocessing
import threading
import cupy
import pytorch

def init_pythtracer():
    """
    Initialize the pathtracer
    """
    # Initialize the camera
    from core.camera_ray import CameraRay
    camera_ray = CameraRay()

    # Initialize the shadows
    from core.shadow_ray import ShadowRay
    shadow_ray = ShadowRay()

    # Initialize the render equation
    from core.render_equation import RenderEquation
    render_equation = RenderEquation()

    # Initialize the recursive tracing
    from core.recursive_tracing import RecursiveTracing
    recursive_tracing = RecursiveTracing()

    # Initialize the intersection shape functions
    from geometry.sphere_intersection import SphereIntersection
    sphere_intersection = SphereIntersection()
    from geometry.plane_intersection import PlaneIntersection
    plane_intersection = PlaneIntersection()
    from geometry.triangle_intersection import TriangleIntersection
    triangle_intersection = TriangleIntersection()
    from geometry.quad_intersection import QuadIntersection
    quad_intersection = QuadIntersection()
    from geometry.cube_intersection import CubeIntersection
    cube_intersection = CubeIntersection() 

    # Initialize the light functions
    from light.point_light import PointLight
    point_light = PointLight()
    from light.area_light import AreaLight
    area_light = AreaLight()
    from light.radiance_unit import RadianceUnit
    radiance_unit = RadianceUnit()

    # Initialize the materials
    from material.diffuse_lambert import DiffuseLambert
    diffuse_lambert = DiffuseLambert()
    from material.mirror import Mirror
    mirror = Mirror()
    from material.glossy import Glossy
    glossy = Glossy()
    from material.glass import Glass
    glass = Glass()

    # Initialize the camera functions
    from camera.camera_exposure import CameraExposure
    camera_exposure = CameraExposure()
    from camera.depth_of_field import DepthOfField
    depth_of_field = DepthOfField()
    from camera.motion_blur import MotionBlur
    motion_blur = MotionBlur()

    # Initialize the optimization functions
    from optimization.multiprocessing import Multiprocessing
    multiprocessing = Multiprocessing()
    from optimization.anti_aliasing import AntiAliasing
    anti_aliasing = AntiAliasing()

    # Initialize the imaging functions
    from imaging.gamma_correction import GammaCorrection
    gamma_correction = GammaCorrection()
    from imaging.tone_mapping import ToneMapping
    tone_mapping = ToneMapping()

    # Initialize the GUI functions
    from gui.render_window import RenderWindow
    render_window = RenderWindow()
    from gui.bucket_render import BucketRender
    bucket_render = BucketRender()
    from gui.real_time_update import RealTimeUpdate
    real_time_update = RealTimeUpdate()
    from gui.separate_thread import SeparateThread
    separate_thread = SeparateThread()
    from gui.render_time_calculation import RenderTimeCalculation
    render_time_calculation = RenderTimeCalculation()
    from gui.interactive_render import InteractiveRender
    interactive_render = InteractiveRender()
    from gui.global_render_settings import GlobalRenderSettings
    global_render_settings = GlobalRenderSettings() 
