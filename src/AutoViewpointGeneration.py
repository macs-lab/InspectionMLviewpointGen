# __________________________________________________________
# IMPORTS
# __________________________________________________________
import os
import sys
import time
import copy
import math
import random
import numpy as np
import pytransform3d.rotations as pr
import open3d as o3d  # . . . . . . . . . . . . . . . Open3D
from open3d.geometry import TriangleMesh, PointCloud
from open3d.visualization.rendering import MaterialRecord
from datetime import datetime
from dataclasses import dataclass, field
from sklearn.cluster import KMeans  # . . . . . . . . K-means
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale
from pathlib import Path
from math import pi, sin, cos
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

@dataclass
class SurfaceModel:
    """ Surface model object. """

    name: str = ''
    folder: str = ''
    file: str = ''
    mesh: TriangleMesh = None
    material: MaterialRecord = MaterialRecord()

    @classmethod
    def load_from_file(self, file, units='mm', color=[1., 1., 1., 1.]):
        """ Load object from file """
        print(f'Loading model: {file}')

        name = ''
        mesh = None
        material = MaterialRecord()
        material.base_color = color
        material.shader = 'defaultLitTransparency'

        # Load model if instantiated with a file path
        if file is not None:

            # If model folder doesn't exist, make folder
            model_path = Path(file)
            model_file = model_path.name
            model_name = str(model_path.stem)
            model_folder = model_path.parent
            model_ext = model_path.suffix
            print(f'folder: {model_folder}, ext: {model_ext}')

            mesh = o3d.io.read_triangle_mesh(file, print_progress=True)

            model_name_folder = model_folder / model_name

            # Compute normals at vertices
            mesh.compute_vertex_normals()
            if model_folder.stem == 'Model':
                model_folder = model_folder.parent
                model_name_folder = model_folder
            elif not model_name_folder.exists():
                os.makedirs(model_name_folder / 'Model')
                file = str(model_name_folder / 'Model' / model_file)
                print(f'File: {file}')
                o3d.io.write_triangle_mesh(file, mesh, print_progress=True)

            # Lookup conversion from units to millimeters
            conversion_table = {'mm': 1,
                                'cm': 10,
                                'm': 1000,
                                'in': 25.4,
                                'ft': 304.8}
            conversion = conversion_table[units]

            # Translate mesh from center in old to new units
            mesh.translate(conversion*mesh.get_center() - mesh.get_center())
            # Scale mesh from original units into new units
            mesh.scale(conversion, mesh.get_center())

            # Compute normals at vertices
            mesh.compute_vertex_normals()

            # Normalize vertex normals
            mesh.normalize_normals()

        return SurfaceModel(model_name, model_name_folder, file, mesh, material)

@dataclass
class SurfacePointCloud:
    """ Surface point cloud object. """

    name: str = ''
    file: str = ''
    pcd: PointCloud = None
    ppsqmm: int = 10
    material: MaterialRecord = MaterialRecord()

    @classmethod
    def load_from_file(self, file, units='mm', ppsqmm=10, color=[0., 1., 0., 1.]):
        """ Load object from file """

        mesh = None
        material = MaterialRecord()
        material.base_color = color
        material.shader = 'defaultLit'

        if file is not None:
            name = os.path.splitext(os.path.basename(file))[0]
            pcd = o3d.io.read_point_cloud(file)

            # Lookup conversion from units to millimeters
            conversion_table = {'mm': 1,
                                'cm': 10,
                                'm': 1000,
                                'in': 25.4,
                                'ft': 304.8}
            conversion = conversion_table[units]

            # Translate mesh from center in old to new units
            pcd.translate(conversion*pcd.get_center() - pcd.get_center())
            # Scale mesh from original units into new units
            pcd.scale(conversion, pcd.get_center())
            pcd.estimate_normals()

        return SurfacePointCloud(name, file, pcd, ppsqmm, material)

    def get_pcd(self):
        return (self.name, copy.deepcopy(self.pcd), self.material)


@dataclass
class Viewpoint:
    """ Defines a coordinate frame. """
    tf: np.array = np.eye(4)

    def get_mesh(self):
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=10).transform(self.tf)
        material = MaterialRecord()
        material.shader = 'defaultLit'
        return mesh, material

    def get_position_orientation(self):
        position = self.tf[:3, 3].reshape((3,)).tolist()
        orientation = pr.quaternion_xyzw_from_wxyz(
            pr.quaternion_from_matrix(self.tf[:3, :3])).tolist()
        return position, orientation

    @classmethod
    def from_dict(self, dict):
        x_hat = np.array(dict['x_hat'])
        y_hat = np.array(dict['y_hat'])
        z_hat = np.array(dict['z_hat'])
        point = np.array(dict['point'])

        tf = np.eye(4)
        tf[:3, 0] = x_hat
        tf[:3, 1] = y_hat
        tf[:3, 2] = z_hat
        tf[:3, 3] = point

        return Viewpoint(tf)

    def to_dict(self):
        return {'x_hat': self.tf[:3, 0].reshape((3,)).tolist(),
                'y_hat': self.tf[:3, 1].reshape((3,)).tolist(),
                'z_hat': self.tf[:3, 2].reshape((3,)).tolist(),
                'point': self.tf[:3, 3].reshape((3,)).tolist()}


@dataclass
class SurfaceRegion:
    """ Surface region object. """

    name: str = ''
    pcd: SurfacePointCloud = SurfacePointCloud()
    mesh: TriangleMesh = TriangleMesh()
    origin: tuple = (0., 0., 0.)
    normal: tuple = (0., 0., 1.)
    viewpoint: Viewpoint = Viewpoint()

    @classmethod
    def load_from_file(self, name, file):
        print(f'Loading region: {file}')
        pcd = SurfacePointCloud.load_from_file(file)
        mesh, _ = pcd.pcd.compute_convex_hull(joggle_inputs=True)
        mesh.compute_vertex_normals()

        return SurfaceRegion(name=name, pcd=pcd, mesh=mesh)

    def get_objs(self, selected=False):
        """ Copy and return all objects associated with surface region """
        objs = []
        # PCD
        (pcd_name, region_pcd, region_material) = self.pcd.get_pcd()
        # objs.append((f'{self.name}_pcd', region_pcd, region_material))
        # Mesh resample
        mesh = copy.deepcopy(self.mesh).translate(0.1*self.normal)
        objs.append((f'{pcd_name}_mesh', mesh, region_material))
        # Create origin mesh

        # Create viewpoint coordinate frame mesh
        viewpoint_mesh, viewpoint_material = self.viewpoint.get_mesh()
        objs.append((f'{self.name}_viewpoint',
                    viewpoint_mesh, viewpoint_material))

        if selected:
            # create selection objects
            viewpoint_outline_mesh = TriangleMesh.create_octahedron(radius=5.0)
            viewpoint_outline_mesh.compute_vertex_normals()
            viewpoint_outline_mesh.transform(self.viewpoint.tf)
            viewpoint_arrow_mesh = TriangleMesh.create_arrow(
                cylinder_height=90-5)
            viewpoint_arrow_mesh.compute_vertex_normals()
            viewpoint_arrow_mesh.transform(self.viewpoint.tf)
            viewpoint_outline_material = MaterialRecord()
            viewpoint_outline_material.shader = 'defaultLitTransparency'
            viewpoint_outline_material.base_color = [1., 1., 1., 0.7]
            objs.append((f'{self.name}_viewpoint_outline',
                        viewpoint_outline_mesh, viewpoint_outline_material))
            objs.append((f'{self.name}_viewpoint_arrow',
                        viewpoint_arrow_mesh, viewpoint_outline_material))

        return objs

    def save(self, folder):
        """ Save pcd to folder. File name is region name. """
        file = folder / Path(self.name + '.ply')
        o3d.io.write_point_cloud(str(file), self.pcd.pcd)
        self.pcd.file = str(file)
        print(str(Path(sys.path[0], str(file))))


@dataclass
class SurfaceResampler:
    """ Service for generating a PCD from TriangleMeshes. """

    points_per_square_mm: int = 10

    def resample(self, model: SurfaceModel()) -> SurfacePointCloud():
        """ Calculate total number of points and perform poisson disk sampling """
        # Creating pcd folder should be handled by auto viewpoint generation app.
        # Move functionality over there eventually.

        # If PCD folder doesn't exist, create it
        model_name_pcd_folder = model.folder / 'PointCloud'
        print(f'folder: {model_name_pcd_folder}')

        if not os.path.exists(model_name_pcd_folder):
            os.makedirs(model_name_pcd_folder)

        name = model.name + '_' + str(self.points_per_square_mm) + '_ppsqmm'
        pcd_file = model_name_pcd_folder / Path(name + '.ply')

        if pcd_file.exists():
            pcd = o3d.io.read_point_cloud(str(pcd_file))
            pcd.estimate_normals()
        else:
            number_of_points = int(
                self.points_per_square_mm * model.mesh.get_surface_area())
            pcd = model.mesh.sample_points_poisson_disk(number_of_points)
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(10)
            # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
            o3d.io.write_point_cloud(str(pcd_file), pcd, print_progress=True)

        material = MaterialRecord()
        material.shader = 'defaultLit'

        return SurfacePointCloud(name, pcd_file, pcd, self.points_per_square_mm, material)


@dataclass
class Camera:
    """ Object for storing camera settings """
    sensor_width_mm: float = 35.7  # mm
    sensor_height_mm: float = 23.8  # mm
    sensor_width_px: int = 9504  # px
    sensor_height_px: int = 6336  # px
    magnification: int = 3
    roi_height: int = 1024  # px
    roi_width: int = 1024  # px
    focal_distance: int = 28  # cm
    dof: int = 4 # mm

    def get_fov(self):
        width = self.magnification * \
            (self.roi_width/self.sensor_width_px)*self.sensor_width_mm
        height = self.magnification * \
            (self.roi_height/self.sensor_height_px)*self.sensor_height_mm
        return width, height

    def get_area(self):
        width, height = self.get_fov()
        return pi*(height/2)**2


@dataclass
class PointCloudPartitioner:
    """ Service for clustering points and returning arrays of points. """

    k: int = 1
    point_weight: float = 1
    normal_weight: float = 1
    initial_method: str = "random"
    maximum_iterations: int = 100
    number_of_runs: int = 10
    bs_high_scaling: int = 2
    k_evaluation_tries: int = 1
    area_per_point = 0
    eval_tries = 0
    min_tries = 0
    pcd_common: PointCloud = PointCloud()
    ppsqmm = 10
    camera_common: Camera = Camera()
    valid_pcds = []
    overall_packing_efficiency = 0
    total_point_out_percentage=0
    b_opt_init_points=1
    b_opt_iteration_points=3
    b_opt_range_multiplier=3
    b_opt_aquisition_function="ei"
    b_opt_lamda_weight: float = 100
    b_opt_beta_weight: float = 2
    

    def evaluate_cluster_dof(self, pcd: PointCloud, camera: Camera):
        """ Function for testing different ways of evaluating validity of a cluster. """

        if not pcd.has_normals():
            pcd.estimate_normals()
        pcd.normalize_normals()

        normals = np.asarray(pcd.normals)
        z = np.array([0, 0, 1])
        z_hat = np.average(normals, 0)
        x_hat = np.cross(z, z_hat)
        y_hat = np.cross(z_hat, x_hat)

        x_hat = x_hat/np.linalg.norm(x_hat)
        y_hat = y_hat/np.linalg.norm(y_hat)
        z_hat = z_hat/np.linalg.norm(z_hat)

        R = np.hstack(
            (x_hat.reshape(3, 1), y_hat.reshape(3, 1), z_hat.reshape(3, 1)))

        pcd.rotate(np.linalg.inv(R), pcd.get_center())
        pcd.translate(-pcd.get_center())
        pcd.estimate_normals()
        pcd.normalize_normals()

        camera_width, camera_height = camera.get_fov()
        camera_r = camera_height/2
        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.zeros((len(pcd_points), 3))

        fov_mesh = TriangleMesh.create_cylinder(
            radius=camera_r, height=camera.dof, resolution=1000, split=1)
        fov = o3d.geometry.LineSet.create_from_triangle_mesh(fov_mesh)
        lines = np.asarray(fov.lines)
        delete = []
        for i in range(len(lines)):
            line = lines[i, :]
            if line[0] <= 1 or line[1] <= 1:
                delete.append(i)
        lines = np.delete(lines, delete, 0)
        fov.lines = o3d.utility.Vector2iVector(lines)
        xyz = TriangleMesh.create_coordinate_frame()
        txt = o3d.t.geometry.TriangleMesh.create_text(
            text=f'''D = {round(2*camera_r, 3)} mm, {int(camera.roi_width)} px''', depth=0, float_dtype=o3d.core.Dtype.Float32)
        txt.scale(0.05, o3d.core.Tensor(np.array([0, 0, 0])))
        p_np = np.array([camera_r, camera_r, 0])
        p = o3d.core.Tensor(p_np)
        txt.translate(p)
        arr = o3d.geometry.LineSet()
        arr.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
        arr.points = o3d.utility.Vector3dVector(np.array([[0.707*camera_r, 0.707*camera_r, 0],
                                                          [camera_r, camera_r, 0]]))

        material = MaterialRecord()
        material.shader = 'defaultLit'
        material.base_color = [1., 1., 1., 1.]

        objs = [('pcd', pcd, material),
                ('fov', fov, material),
                ('xyz', xyz, material),
                ('txt', txt, material),
                ('arr', arr, material)]

        valid = True

        try:
            red_count = 0
            green_count = 0
            z_max = 0
            z_min = 0
            for i in range(len(pcd_points)):
                p = pcd_points[i, :]
                x, y, z = p[0], p[1], p[2]
                z_max = z if z > z_max else z_max
                z_min = z if z < z_min else z_min
                if abs(z) > camera.dof/2:
                    pcd_colors[i, 0] = 1.  # paint red
                    red_count += 1

                else:
                    pcd_colors[i, 1] = 1.  # paint green
                    green_count += 1
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
        except Exception as e:
            print(e)
        if (red_count/(red_count+green_count) < 0.050):
            valid = True
        else:
            valid = False

        # cost = (x_max - x_min)*(y_max - y_min)
    
        point_out_percentage = red_count/(green_count+red_count)
        max_height = camera.dof
        point_height=abs(z_max-z_min)
        packing_eff = point_height/max_height
        
        return valid,point_out_percentage,packing_eff, objs    # return valid, cost, green_count, objs

    def evaluate_cluster_fov(self, pcd: PointCloud, camera: Camera):
        """ Function to be implemented for testing different ways of evaluating validity of a cluster. """
        if not pcd.has_normals():
            pcd.estimate_normals()
        pcd.normalize_normals()

        normals = np.asarray(pcd.normals)
        z = np.array([0, 0, 1])
        z_hat = np.average(normals, 0)
        # if(np.linalg.norm(z_hat==0)):
        #     print("zhat is zero")
        x_hat = np.cross(z, z_hat)
        if (np.linalg.norm(x_hat) == 0):
            # print("x_hat is zero")
            # print(z_hat)
            x_hat = np.array([z_hat[2], 0, 0])
            print(x_hat)
        y_hat = np.cross(z_hat, x_hat)
        if (np.linalg.norm(y_hat) == 0):
            # print("y_hat is zero")
            # print(x_hat, z_hat)
            y_hat = np.array([0, 1, 0])
        x_hat = x_hat/np.linalg.norm(x_hat)
        y_hat = y_hat/np.linalg.norm(y_hat)
        z_hat = z_hat/np.linalg.norm(z_hat)

        R = np.hstack(
            (x_hat.reshape(3, 1), y_hat.reshape(3, 1), z_hat.reshape(3, 1)))

        pcd.rotate(np.linalg.inv(R), pcd.get_center())
        pcd.translate(-pcd.get_center())
        pcd.estimate_normals()
        pcd.normalize_normals()
        # temp_pcd_hull, _ = pcd.compute_convex_hull()

        camera_width, camera_height = camera.get_fov()
        camera_r = camera_height/2
        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.zeros((len(pcd_points), 3))

        fov_mesh = TriangleMesh.create_cylinder(
            radius=camera_r, height=camera.dof, resolution=1000, split=1)
        fov = o3d.geometry.LineSet.create_from_triangle_mesh(fov_mesh)
        lines = np.asarray(fov.lines)
        delete = []
        for i in range(len(lines)):
            line = lines[i, :]
            if line[0] <= 1 or line[1] <= 1:
                delete.append(i)
        lines = np.delete(lines, delete, 0)
        fov.lines = o3d.utility.Vector2iVector(lines)
        xyz = TriangleMesh.create_coordinate_frame()
        txt = o3d.t.geometry.TriangleMesh.create_text(
            text=f'''D = {round(2*camera_r, 3)} mm, {int(camera.roi_width)} px''', depth=0, float_dtype=o3d.core.Dtype.Float32)
        txt.scale(0.05, o3d.core.Tensor(np.array([0, 0, 0])))
        p_np = np.array([camera_r, camera_r, 0])
        p = o3d.core.Tensor(p_np)
        txt.translate(p)
        arr = o3d.geometry.LineSet()
        arr.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
        arr.points = o3d.utility.Vector3dVector(np.array([[0.707*camera_r, 0.707*camera_r, 0],
                                                          [camera_r, camera_r, 0]]))

        material = MaterialRecord()
        material.shader = 'defaultLit'
        material.base_color = [1., 1., 1., 1.]

        objs = [('pcd', pcd, material),
                ('fov', fov, material),
                ('xyz', xyz, material),
                ('txt', txt, material),
                ('arr', arr, material)]

        valid = True

        try:
            red_count = 0
            green_count = 0
            x_max = 0
            x_min = 100000000
            y_max = 0
            y_min = 100000000
            x_max_in = 0
            x_min_in = 100000000
            y_max_in = 0
            y_min_in = 100000000
            extreme_point_length = 0
            for i in range(len(pcd_points)):
                p = pcd_points[i, :]
                x, y, z = p[0], p[1], p[2]
                # x_max = x if x > x_max else x_max
                # x_min = x if x < x_min else x_min
                # y_max = y if y > y_max else y_max
                # y_min = y if y < y_min else y_min
                extreme_point_length = math.sqrt(x**2 + y**2) if math.sqrt(x**2 + y**2) > extreme_point_length else extreme_point_length
                if math.sqrt(x**2 + y**2) > camera_r:
                    pcd_colors[i, 0] = 1.  # paint red
                    red_count += 1

                else:
                    x_max_in = x if x > x_max_in else x_max_in
                    x_min_in = x if x < x_min_in else x_min_in
                    y_max_in = y if y > y_max_in else y_max_in
                    y_min_in = y if y < y_min_in else y_min_in
                    pcd_colors[i, 1] = 1.  # paint green
                    green_count += 1
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
        except Exception as e:
            print(e)
        if self.eval_tries == 0:
            self.area_per_point = 1/self.ppsqmm
            print(self.area_per_point)

            self.eval_tries = 1
        # print(self.area_per_point*(red_count+green_count))

        if (red_count/(red_count+green_count) < 0.003):
            valid = True
        else:
            valid = False

        point_out_percentage = red_count/(green_count+red_count)
        max_points_in = (pi*camera_r**2)/self.area_per_point
        packing_eff = green_count/max_points_in
        borderline=0
        if (extreme_point_length < (camera_r*1.05) and extreme_point_length > (camera_r*0.95)) or (extreme_point_length < (camera_r*1.05) and extreme_point_length > (camera_r*0.95)) and valid==True:
            borderline=1
        return valid,point_out_percentage,packing_eff, objs, borderline
        ##cost 3
       

    def partition(self, pcd: PointCloud(), k: int) -> list:
        """ K-Means clustering function. """

        # Unpack and scale vertices and normalize normals from PCD
        if not pcd.has_normals():
            pcd.estimate_normals()

        pcd.normalize_normals()  # Normals must be normalized for correct clustering

        normals = np.asarray(pcd.normals)
        points = np.asarray(pcd.points)

        if (self.normal_weight != 0):
            # Scale point locations to lie between [-1, 1]
            points = 2 * (minmax_scale(points) - 0.5)

        # Combine weighted vertex and normal data
        data = np.concatenate((self.point_weight * points,
                               self.normal_weight * normals), axis=1)
        # data = normals

        # Scikit Learn KMeans
        KM = KMeans(init='k-means++',
                    n_clusters=k,
                    n_init=self.number_of_runs,
                    max_iter=self.maximum_iterations)
        KM.fit(data)

        labels = KM.labels_
        cluster_collection = [[] for i in range(k)]

        for j in range(len(labels)):
            cluster_collection[labels[j]].append(j)

        # List stores regions
        pcds = []
        for i in range(k):
            pcd_i = copy.deepcopy(pcd.select_by_index(cluster_collection[i]))
            pcds.append(pcd_i)

        return pcds

    def evaluate_k(self, pcd: PointCloud, k: int, camera: Camera, eval_fun: callable, tries: int = 1) -> bool:
        """ Run multiple k-means partitions to determine if current k is valid. """
        for i in range(tries):
            pcds = self.partition(copy.deepcopy(pcd), k)
            k_valid = True
            for j, pcd_1 in enumerate(pcds):
                cluster_valid, cost,_, _ = eval_fun(copy.deepcopy(pcd_1), camera)
                k_valid = k_valid and cluster_valid
                # print(f'k-{k} pcd {j}: {cluster_valid}')
                if not cluster_valid:
                    break
            if k_valid:
                return True, pcds, cost
        return False, pcds, cost

    def evaluate_k_cost_filter(self, k):
        """ Calls partitioning service to partition surface into planar patches then regions. """
        self.regions = []
        # non_valid_pcd, cost, pcds = self.pcd_partitioner.evaluate_k_cost(copy.deepcopy(
        #     self.pcd.pcd), k, self.camera, self.pcd_partitioner.evaluate_cluster_fov, tries=1)
        cost = self.evaluate_k_cost(copy.deepcopy(
            self.pcd_common), k, self.camera_common, self.evaluate_cluster_fov, tries=1)
        return cost





    def evaluate_k_cost(self, pcd: PointCloud, k: float, camera: Camera, eval_fun: callable, tries: int = 1) -> bool:
        #this is to make sure that k is always an integer and minimum value of k is 1
        k=max(1,int(k))
        for i in range(tries):
            pcds = self.partition(copy.deepcopy(pcd), k)
            k_valid = True
            total_cost = 0
            non_valid_pcd = 0
            total_count=0
            total_point_out_percentage=0
            total_packing_eff=0
            anyborderline=False
           
            for j, pcd_1 in enumerate(pcds):
                cluster_valid, point_out_percentage,packing_eff, _ ,borderline= eval_fun(copy.deepcopy(pcd_1), camera)
                total_point_out_percentage += point_out_percentage
                total_packing_eff+=packing_eff
                total_count+=1
                # print(f'k-{k} pcd {j}: {cluster_valid}')
                if not cluster_valid:
                    non_valid_pcd += 1
                if(borderline==1):
                    anyborderline=True
                    # print("border line found")
            total_point_out_percentage=total_point_out_percentage/total_count
            total_packing_eff=total_packing_eff/total_count


            initial_beta=self.b_opt_beta_weight
            
            if(total_point_out_percentage>0.001):
                # print(total_point_out_percentage)
                s=0
            else:
                s=1
                if(anyborderline==True):
                     s=0
                # print(total_packing_eff)
                
            total_cost = (self.b_opt_lamda_weight)*total_point_out_percentage + s*((1/total_packing_eff)**initial_beta)
            
        return -total_cost

    def optimize_k(self, pcd: PointCloud, camera: Camera, eval_fun: callable, bs_high: int = 1) -> int:
        """ Function to perform K-Means binary search with evaluation to determine optimal number of clusters for inspection """
        print('Finding K upper bound...')
        while (True):
            valid, pcd_1, cost = self.evaluate_k(
                copy.deepcopy(pcd), bs_high, camera, eval_fun, tries=1)
            if not valid:
                print(
                    f'K_high = {bs_high} is not valid, incrementing K_high...')
                bs_high *= 2
            else:
                print(f'K_high = {bs_high} is valid. Starting Binary search between K_high/2 and K_high...')
                break
        bs_mid = 0
        bs_low = max(bs_high//2, 1)
        valid_pcds = pcd_1
        while (bs_high > bs_low):
            bs_mid = (bs_low + bs_high)//2
            k = bs_mid
            valid, pcds,_= self.evaluate_k(
                copy.deepcopy(pcd), k, camera, eval_fun)
            print(f'K: {k}, Valid: {valid}')
            if not valid:
                bs_low = bs_mid + 1
            else:
                valid_pcds = pcds
                bs_high = bs_mid
        print(f'K: {bs_high} is valid for DOF K-means clustering.')

        k = bs_high

        return k, valid_pcds
    

    def optimize_k_b_opt(self, pcd: PointCloud, camera: Camera, eval_fun: callable, bs_high: int = 1) -> int:
        try:

            self.min_tries=0
            self.eval_tries=0
            temp_mesh, _ = pcd.compute_convex_hull(joggle_inputs=True)
            area=temp_mesh.get_surface_area()/2
            print("total area of planar segment", area)
            n_est = area/camera.get_area()
            print("K_min",n_est)
            pbounds={"k":(n_est,self.b_opt_range_multiplier*n_est)}

            optimizer=BayesianOptimization(
                        f=self.evaluate_k_cost_filter,
                        pbounds=pbounds,
                        verbose=2, #verbose=1 prints only at max, verbose=0 is silent
                        random_state=1,    
                    )
            acq_function = UtilityFunction(kind=self.b_opt_aquisition_function, kappa=5)
            optimizer.maximize(
                init_points=self.b_opt_init_points,
                n_iter=self.b_opt_iteration_points,  
                )
            y=optimizer.max
            k=max(1,int(y["params"]["k"]))
            print(k)
            valid_pcds = self.partition(copy.deepcopy(pcd), k)
            non_valid_pcd = 0
            total_count=0
            total_point_out_percentage=0
            total_packing_eff=0
           
            for j, pcd_1 in enumerate(valid_pcds):
                cluster_valid, point_out_percentage,packing_eff, _,borderline = eval_fun(copy.deepcopy(pcd_1), camera)
                total_point_out_percentage += point_out_percentage
                total_packing_eff+=packing_eff
                total_count+=1
                # print(f'k-{k} pcd {j}: {cluster_valid}')
                if not cluster_valid:
                    non_valid_pcd += 1
            total_point_out_percentage=total_point_out_percentage/total_count
            total_packing_eff=total_packing_eff/total_count
            # print(total_packing_eff,"total packing efficiency for given planar_pcd")
            self.overall_packing_efficiency+=total_packing_eff
            self.total_point_out_percentage+=total_point_out_percentage

        except Exception as e:
             print(e)

        return k, valid_pcds

   
# Region growing added to the smart partition function
    def smart_partition(self, spcd: SurfacePointCloud, camera: Camera) -> list:
        """ Partition PCD into Planar Patches, partition Planar Patches into Regions. """
        print(f'Partitioning part into planar patches:')
        # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10,
        #                                                 std_ratio=0.01)
        # pcd= pcd.select_by_index(ind, invert=True)
        self.overall_packing_efficiency=0
        self.total_point_out_percentage=0
        self.camera_common=copy.deepcopy(camera)
        self.ppsqmm = spcd.ppsqmm
        theta_th = 4.0 / 180.0 * math.pi  # in radians
        cur_th = 0.01
        num_nieghbors = 30
        pcd = spcd.pcd
        rg_regions = []
        region_pcds = []
        self.total_planar_pcds = 0

        #store point cloud as numpy array
        unique_rows = np.asarray(pcd.points)
        #Generate a KDTree object
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        search_results = []

        #search for 10 nearest neighbors for each point in the point cloud and store the k value, index of the nearby points and their distances them in search_results
        for point in pcd.points:
            try:
                result = pcd_tree.search_knn_vector_3d(point, num_nieghbors)
                search_results.append(result)
            except RuntimeError as e:
                print(f"An error occurred with point {point}: {e}")
                continue

        #separate the k and index values from the search_results

        k_values = [result[0] for result in search_results]
        nn_glob = [result[1] for result in search_results]
        distances = [result[2] for result in search_results]

        region1 = self.regiongrowing1(unique_rows, nn_glob, theta_th, cur_th)
        #visualize the region with each region having different color
        colors = np.random.rand(len(region1),3)  #generating random colors
        pcd_colors = np.zeros((len(pcd.points), 3))  #initializing the color array 
        #region stored
        # colour all regions with points less than 1000 with grey and remove them

        initial_normal_weight = self.normal_weight
        for i in range(len(region1)):
            if len(region1[i]) < 0.005*len(unique_rows):
                continue
            else:
                pcd_i = copy.deepcopy(pcd.select_by_index(region1[i]))
                rg_regions.append(pcd_i)
        for j,rg_region in enumerate(rg_regions):
            k_dof, planar_pcds = self.optimize_k(
                copy.deepcopy(rg_region), camera, self.evaluate_cluster_dof)
            region_planar_pcds = []
            
            self.normal_weight = 0
            for i, planar_pcd in enumerate(planar_pcds):

                print(f'Partitioning planar patch {i} into regions:')
                self.pcd_common=copy.deepcopy(planar_pcd)
                k_roi, pcds = self.optimize_k_b_opt(copy.deepcopy(
                    planar_pcd), camera, self.evaluate_cluster_fov)
                self.total_planar_pcds += 1
                region_planar_pcds += copy.deepcopy(pcds)
            region_pcds.extend(region_planar_pcds)
        self.normal_weight = initial_normal_weight
        total_packing_efficiency=self.overall_packing_efficiency/self.total_planar_pcds
        total_point_out_percentage=self.total_point_out_percentage/self.total_planar_pcds
        print("overall packing efficiency",total_packing_efficiency)
        print("total point out percentage",total_point_out_percentage)
        return region_pcds

    def display_inlier_outlier(self, cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw([inlier_cloud, outlier_cloud])


@dataclass
class ViewpointGenerator:
    """ Service for generating viewpoints from surface regions. """

    def generate_viewpoint(self, region: SurfaceRegion(), camera: Camera()) -> SurfaceRegion:
        """ Takes in a list of PCDs and returns a list of Region objects with viewpoints """

        pcd = region.pcd.pcd
        origin = pcd.get_center()
        if not pcd.has_normals():
            pcd.estimate_normals()
        pcd.normalize_normals()

        normals = np.asarray(pcd.normals)
        normal = np.average(normals, 0)
        normal = normal/np.linalg.norm(normal)

        p = origin + 10*camera.focal_distance*normal

        z = np.array([0, 0, 1])
        z_hat = -normal
        x_hat = np.cross(z, z_hat)
        if (np.linalg.norm(x_hat) == 0):
            x_hat = np.array([z[2], 0, 0])
        y_hat = np.cross(z_hat, x_hat)

        x_hat = x_hat/np.linalg.norm(x_hat)
        y_hat = y_hat/np.linalg.norm(y_hat)
        z_hat = z_hat/np.linalg.norm(z_hat)

        tf = np.eye(4)
        tf[:3, 0] = x_hat
        tf[:3, 1] = y_hat
        tf[:3, 2] = z_hat
        tf[:3, 3] = p

        region.viewpoint = Viewpoint(tf)
        region.origin = origin
        region.normal = normal

        return region

    def adjust_viewpoint(self, region: SurfaceRegion(), model: SurfaceModel(), camera: Camera()):
        """ Takes in a Region object and a Model and adjusts the viewopint such that all points are visible """
        print('Adjusting viewpoint...')

        def check_viewpoint(region: SurfaceRegion(), model: SurfaceModel(), viewpoint: np.ndarray = None):
            model_mesh = model.mesh
            if not type(model_mesh) == type(o3d.t.geometry.TriangleMesh()):
                model_mesh = o3d.t.geometry.TriangleMesh.from_legacy(
                    model_mesh)  # turn object to tensor-based geometry
            # Create raycasting scene
            scene = o3d.t.geometry.RaycastingScene()
            mesh_id = scene.add_triangles(
                model_mesh)  # add model mesh to scene

            # Iterate through points in pcd, creating and accumulating rays
            if viewpoint is None:
                viewpoint = region.viewpoint.tf[:3, 3].reshape((3,))
            x, y, z = viewpoint[0], viewpoint[1], viewpoint[2]

            rays = []
            dists = []
            for i, surface_point in enumerate(region.pcd.pcd.points):
                dist = np.linalg.norm(surface_point - viewpoint)
                dists.append(dist)
                dir = (surface_point - viewpoint)/dist
                u, v, w = dir[0], dir[1], dir[2]
                ray = [x, y, z, u, v, w]
                rays.append(ray)

            # Cast rays in scene
            rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

            ans = scene.cast_rays(rays)
            t_hits = ans['t_hit'].numpy().tolist()
            visible = True
            for i, t_hit in enumerate(t_hits):
                dist = dists[i]
                if abs(1 - t_hit/dist) > 0.01:
                    visible = False

            # Iterate over rays, checking distances to first hit against ideal distances.
            sigma = np.std(ans['t_hit'].numpy())

            return visible, sigma

        z = np.array([0, 0, 1])

        z_hat = region.normal
        x_hat = np.cross(z, z_hat)
        y_hat = np.cross(z_hat, x_hat)

        x_hat = x_hat/np.linalg.norm(x_hat)
        y_hat = y_hat/np.linalg.norm(y_hat)
        z_hat = z_hat/np.linalg.norm(z_hat)

        T = np.eye(4)
        T[:3, 0] = x_hat
        T[:3, 1] = y_hat
        T[:3, 2] = z_hat
        T[:3, 3] = region.origin

        offset = 10*camera.focal_distance

        visible, sigma = check_viewpoint(
            region, model, viewpoint=T.dot(np.array([0, 0, offset, 1]))[:3])
        if visible:
            return

        phi_max = pi/2
        phi_count = 10
        l = 2*pi*offset/phi_count

        # Count number of viewpoint samples
        tot = 0
        for phi in np.linspace(0, phi_max, phi_count):
            theta_count = max(1, int(2*pi*sin(phi)*offset/l))
            tot = tot + theta_count

        # Compile viewpoints in numpy array and check viewpoint
        viewpoints = np.zeros((tot, 3))
        visible_indices = []
        visible_sigmas = []
        count = 0
        for phi in np.linspace(0, phi_max, phi_count):
            theta_count = max(1, int(2*pi*sin(phi)*offset/l))
            for i in range(theta_count):
                theta = 2*pi*random.random()
                x = offset*cos(theta)*sin(phi)
                y = offset*sin(theta)*sin(phi)
                z = offset*cos(phi)
                v_hat = np.array([x, y, z, 1])
                v = T.dot(v_hat)[:3]
                viewpoints[count, :] = v

                t0 = time.time()
                visible, sigma = check_viewpoint(region, model, viewpoint=v)
                t1 = time.time()

                if visible:
                    visible_indices.append(count)
                    visible_sigmas.append(sigma)

                count += 1

        if visible_indices:
            visible_idx = np.argmin(np.array(visible_sigmas))
            idx = visible_indices[visible_idx]
            viewpoint = viewpoints[idx, :]

            d = np.linalg.norm(viewpoint - region.origin)

            z = np.array([0, 0, 1])
            z_hat = region.origin - viewpoint
            x_hat = np.cross(z, z_hat)
            y_hat = np.cross(z_hat, x_hat)

            x_hat = x_hat/np.linalg.norm(x_hat)
            y_hat = y_hat/np.linalg.norm(y_hat)
            z_hat = z_hat/np.linalg.norm(z_hat)

            v_tf = np.eye(4)
            v_tf[:3, 0] = x_hat
            v_tf[:3, 1] = y_hat
            v_tf[:3, 2] = z_hat
            v_tf[:3, 3] = viewpoint

            region.viewpoint.tf = v_tf


@dataclass
class ViewpointTraversalOptimizer:
    """ Service for re-ordering a list of viewpoints to minimize traversal time. """

    def sort_viewpoints(self, regions, sides):
        """ take in a list of viewpoints/regions and will sort them and return an ordered list. """
        for side, indices in sides.items():
            if not indices:
                continue
            print(f'Side: {side}')
            print(f'/tunsorted: {indices}')
            waypoint_xyz = []
            sorted_indices_viewpoint = []
            # for i in range(len(indices)):
            for i in indices:
                viewpoint = regions[i].viewpoint.tf[:3, 3].reshape((3,))
                waypoint_xyz.append(viewpoint)
            new_waypoints = np.vstack(waypoint_xyz)
            if side == "front":
                sorted_indices = np.lexsort(
                    (new_waypoints[:, 2], new_waypoints[:, 1]))
            if side == "back":
                sorted_indices = np.lexsort(
                    (new_waypoints[:, 2], -new_waypoints[:, 1]))
            if side == "left":
                sorted_indices = np.lexsort(
                    (new_waypoints[:, 2], new_waypoints[:, 0]))
            if side == "right":
                sorted_indices = np.lexsort(
                    (new_waypoints[:, 2], -new_waypoints[:, 0]))
            if side == "bottom":
                sorted_indices = np.lexsort(
                    (new_waypoints[:, 0], -new_waypoints[:, 1]))
            if side == "top":
                sorted_indices = np.lexsort(
                    (-new_waypoints[:, 0], new_waypoints[:, 1]))
            for j in range(len(sorted_indices)):
                current_index = indices[sorted_indices[j]]
                sorted_indices_viewpoint.append(current_index)
            sides[side] = sorted_indices_viewpoint
            print(f'/tsorted: {sorted_indices_viewpoint}')
        return sides




@dataclass
class AutoViewpointGeneration:
    """ Application for generating evenly distributed viewpoints across the surface of a model. """

    # Geometry Objects
    model: SurfaceModel = SurfaceModel()
    spcd: SurfacePointCloud = SurfacePointCloud()
    regions: list = field(default_factory=list)
    camera: Camera = Camera()

    # Service Objects
    surface_resampler: SurfaceResampler = SurfaceResampler()
    pcd_partitioner: PointCloudPartitioner = PointCloudPartitioner()
    viewpoint_generator: ViewpointGenerator = ViewpointGenerator()
    viewpoint_sorter: ViewpointTraversalOptimizer = ViewpointTraversalOptimizer()

    sides: dict = field(default_factory=dict)
    selected_side: str = None

    def load_model(self, file, units, color):
        """ Function to reset app and load a new model. """
        print(f'Loading model from {file}')
        self.model = SurfaceModel.load_from_file(file, units, color)
        self.spcd = SurfacePointCloud()
        self.regions = []
        self.selected_side = None

    def load_pcd(self, file, units, ppsqmm, color):
        """ Function to load a point cloud for partitioning. """
        print(f'Loading pcd from {file}')
        self.spcd = SurfacePointCloud.load_from_file(file, units, ppsqmm, color)
        self.regions = []
        self.selected_side = None

    def save_regions(self):
        """ Save region pcds. """
        folder = self.model.folder / 'Regions' / \
            datetime.now().strftime('%Y%m%d%H%M%S')
        if not folder.exists():
            os.makedirs(folder)
        for region in self.regions:
            region.save(folder)
#

    def load_regions(self, settings):
        """ Function to load surface regions """
        region_description_list = settings.region_description_list

        self.regions = []
        for region_description in region_description_list:
            # Unpack region descriptions and create new SurfaceRegion object
            name = region_description['name']
            file = str(settings.pkg_root / Path(region_description['file']))
            viewpoint_dict = region_description['viewpoint_tf']
            region = SurfaceRegion.load_from_file(name, file)
            region.origin = np.array(region_description['origin'])
            region.normal = np.array(region_description['normal'])
            region.viewpoint = Viewpoint.from_dict(viewpoint_dict)
            self.regions.append(region)
        self.paint_regions("plasma")
        self.cluster_viewpoints()
        print("Done loading regions.")

    def generate_pcd(self):
        """ Calls resampler service to generate surface pcd """
        self.spcd = self.surface_resampler.resample(self.model)

    def optimize_k(self):
        """ Calls partitioning service to optmize k for partitioning """
        print("App optimizing k")
        return self.pcd_partitioner.optimize_k(self.spcd.pcd, self.camera, self.pcd_partitioner.evaluate_cluster_dof)

    def smart_partition(self):
        """ Calls partitioning service to partition surface into planar patches then regions. """
        # self.regions = []
        pcds = self.pcd_partitioner.smart_partition(self.spcd, self.camera)
        print(pcds)
        return pcds
        # self.pcds_to_regions(pcds)

    def evaluate_k_cost(self, k):
        """ Calls partitioning service to partition surface into planar patches then regions. """
        self.regions = []
        # non_valid_pcd, cost, pcds = self.pcd_partitioner.evaluate_k_cost(copy.deepcopy(
        #     self.pcd.pcd), k, self.camera, self.pcd_partitioner.evaluate_cluster_fov, tries=1)
        cost = self.pcd_partitioner.evaluate_k_cost(copy.deepcopy(
            self.spcd.pcd), k, self.camera, self.pcd_partitioner.evaluate_cluster_fov, tries=1)
        return cost

    def partition(self, k):
        """ Calls partitioning service to generate surface region pcd's """
        self.regions = []
        pcds = self.pcd_partitioner.partition(self.spcd.pcd, k)
        self.pcds_to_regions(pcds)

    def pcds_to_regions(self, pcds):
        for i, pcd in enumerate(pcds):

            origin = pcd.get_center()
            normal = (0., 0., 1.)  # Calculate this
            mesh, _ = pcd.compute_convex_hull(joggle_inputs=True)
            mesh.compute_vertex_normals()
            # mesh=TriangleMesh()
            material = MaterialRecord()
            material.shader = 'defaultLit'
            spcd = SurfacePointCloud(
                name=f'region_{i}_pcd', file='', pcd=pcd, material=material)
            region = SurfaceRegion(f'region_{i}', spcd, mesh, origin, normal)
            self.regions.append(region)
        #hardcoded for now on the color
        self.paint_regions("plasma")
        self.generate_viewpoints()

    def generate_viewpoints(self):
        """ Calls viewpoint generation service to generate non-adjusted viewpoints for all regions. """
        if self.regions:
            for region in self.regions:
                region = self.viewpoint_generator.generate_viewpoint(
                    region, self.camera)
        self.cluster_viewpoints()

    def get_viewpoints_bb(self):
        if self.regions:
            np_points = np.zeros((len(self.regions), 3))
        elif self.model:
            bb = self.model.mesh.get_axis_aligned_bounding_box()
            bb.scale(1.5, bb.get_center())
            return bb
        else:
            np_points = np.array([[1, 0, 0], [-1, 0, 0],
                                  [0, 1, 0], [0, -1, 0],
                                  [0, 0, 1], [0, 0, -1]])
        for i, region in enumerate(self.regions):
            np_points[i, :] = region.viewpoint.tf[:3, 3].reshape((3,))
        # sides_bb = self.model.get_axis_aligned_bounding_box()
        return o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np_points))

    def adjust_viewpoint(self, idx):
        if self.regions:
            self.viewpoint_generator.adjust_viewpoint(
                self.regions[idx], self.model, self.camera)

    def adjust_viewpoints(self):
        if self.selected_side:
            for i in self.sides[self.selected_side]:
                self.adjust_viewpoint(i)
        else:
            for i in range(len(self.regions)):
                self.adjust_viewpoint(i)
        self.cluster_viewpoints()

    def cluster_viewpoints(self):
        """ Clusters viewpoints into front, back, left, right, top, bottom sections. """
        if not self.regions:
            return

        self.sides['front'] = []
        self.sides['right'] = []
        self.sides['back'] = []
        self.sides['left'] = []
        self.sides['top'] = []
        self.sides['bottom'] = []

        vbb = self.get_viewpoints_bb()
        vbbc = vbb.get_center()
        max_bound = vbb.max_bound
        min_bound = vbb.min_bound
        x_max = max_bound[0]
        y_max = max_bound[1]
        z_max = max_bound[2]
        x_min = min_bound[0]
        y_min = min_bound[1]
        z_min = min_bound[2]

        front_p = np.array([x_max, vbbc[1], vbbc[2]])
        front_n = np.array([-1, 0, 0])

        right_p = np.array([vbbc[0], y_max, vbbc[2]])
        right_n = np.array([0, -1, 0])

        back_p = np.array([x_min, vbbc[1], vbbc[2]])
        back_n = np.array([1, 0, 0])

        left_p = np.array([vbbc[0], y_min, vbbc[2]])
        left_n = np.array([0, 1, 0])

        top_p = np.array([vbbc[0], vbbc[1], z_max])
        top_n = np.array([0, 0, -1])

        bottom_p = np.array([vbbc[0], vbbc[1], z_min])
        bottom_n = np.array([0, 0, 1])

        for i, region in enumerate(self.regions):
            p = region.viewpoint.tf[:3, 3].reshape((3,))
            z_hat = region.viewpoint.tf[:3, 2].reshape((3,))
            linear_differences = np.array([np.linalg.norm(p-front_p),
                                           np.linalg.norm(p-right_p),
                                           np.linalg.norm(p-back_p),
                                           np.linalg.norm(p-left_p),
                                           np.linalg.norm(p-top_p),
                                           np.linalg.norm(p-bottom_p)])
            rotation_differences = np.array([np.linalg.norm(np.cross(z_hat, front_n)),
                                             np.linalg.norm(
                                                 np.cross(z_hat, right_n)),
                                             np.linalg.norm(
                                                 np.cross(z_hat, back_n)),
                                             np.linalg.norm(
                                                 np.cross(z_hat, left_n)),
                                             np.linalg.norm(
                                                 np.cross(z_hat, top_n)),
                                             np.linalg.norm(np.cross(z_hat, bottom_n))])
            linear_differences_normed = linear_differences / \
                np.linalg.norm(linear_differences)
            rotation_differences_normed = rotation_differences / \
                np.linalg.norm(rotation_differences)
            min_idx = np.argmin(linear_differences)
            if i == 0:
                pass
            if min_idx == 0:
                self.sides['front'].append(i)
            elif min_idx == 1:
                self.sides['right'].append(i)
            elif min_idx == 2:
                self.sides['back'].append(i)
            elif min_idx == 3:
                self.sides['left'].append(i)
            elif min_idx == 4:
                self.sides['top'].append(i)
            elif min_idx == 5:
                self.sides['bottom'].append(i)

    def sort_viewpoints(self):
        """ Passes list of regions into viewpoint traversal optimizer to sort viewpoints. """
        self.sides = self.viewpoint_sorter.sort_viewpoints(
            self.regions, self.sides)

    def select_side_by_coord(self, coord):
        if not self.regions:
            return

        x, y, z = coord[0], coord[1], coord[2]

        vbb = self.get_viewpoints_bb()
        max_bound = vbb.max_bound
        min_bound = vbb.min_bound
        x_max = max_bound[0]
        y_max = max_bound[1]
        z_max = max_bound[2]
        x_min = min_bound[0]
        y_min = min_bound[1]
        z_min = min_bound[2]
        tol = 5

        if abs(x - x_max) < tol:
            self.selected_side = 'front'
        elif abs(x - x_min) < tol:
            self.selected_side = 'back'
        elif abs(y - y_max) < tol:
            self.selected_side = 'right'
        elif abs(y - y_min) < tol:
            self.selected_side = 'left'
        elif abs(z - z_max) < tol:
            self.selected_side = 'top'
        elif abs(z - z_min) < tol:
            self.selected_side = 'bottom'
        else:
            self.selected_side = None

    def paint_regions(self, colormap):
        cmap = colormaps[colormap]
        for i, region in enumerate(self.regions):
            region.pcd.material.base_color = list(cmap(i/len(self.regions)))

    def get_region_by_index(self, idx):
        if self.regions:
            if self.selected_side:
                return self.regions[self.sides[self.selected_side][idx]]
            else:
                return self.regions[idx]
        else:
            return None

    def get_region_indices(self):
        if self.regions:
            if self._app.selected_side:
                region_idx_list = self.sides[self.selected_side]
            else:
                region_idx_list = list(range(len(self.regions)))

            return region_idx_list
        else:
            return None

    def get_objects_in_scene(self, settings):
        model_obj = self.model.get_mesh()

        # Sides
        sides_bb = self.get_viewpoints_bb()
        max_bound = sides_bb.max_bound
        min_bound = sides_bb.min_bound
        x_max = max_bound[0]
        y_max = max_bound[1]
        z_max = max_bound[2]
        x_min = min_bound[0]
        y_min = min_bound[1]
        z_min = min_bound[2]
        scale = 0.9
        d = 1
        model_c = sides_bb.get_center()
        front_mesh = TriangleMesh.create_box(width=d,
                                             height=scale*(y_max - y_min),
                                             depth=scale*(z_max - z_min))
        front_mesh.translate(-front_mesh.get_center())
        front_mesh.translate(np.array([x_max, model_c[1], model_c[2]]))
        sides_material = MaterialRecord()
        sides_material.base_color = [1., 1., 1., 0.005]
        sides_material.shader = 'defaultLitTransparency'
        front_obj = ('front_side', front_mesh, sides_material)

        right_mesh = TriangleMesh.create_box(width=scale*(x_max - x_min),
                                             height=d,
                                             depth=scale*(z_max - z_min))
        right_mesh.translate(-right_mesh.get_center())
        right_mesh.translate(np.array([model_c[0], y_max, model_c[2]]))
        right_obj = ('right_side', right_mesh, sides_material)

        back_mesh = TriangleMesh.create_box(width=d,
                                            height=scale*(y_max - y_min),
                                            depth=scale*(z_max - z_min))
        back_mesh.translate(-back_mesh.get_center())
        back_mesh.translate(np.array([x_min, model_c[1], model_c[2]]))
        back_obj = ('back_side', back_mesh, sides_material)

        left_mesh = TriangleMesh.create_box(width=scale*(x_max - x_min),
                                            height=d,
                                            depth=scale*(z_max - z_min))
        left_mesh.translate(-left_mesh.get_center())
        left_mesh.translate(np.array([model_c[0], y_min, model_c[2]]))
        left_obj = ('left_side', left_mesh, sides_material)

        top_mesh = TriangleMesh.create_box(width=scale*(x_max - x_min),
                                           height=scale*(y_max - y_min),
                                           depth=d)
        top_mesh.translate(-top_mesh.get_center())
        top_mesh.translate(np.array([model_c[0], model_c[1], z_max]))
        top_obj = ('top_side', top_mesh, sides_material)

        bottom_mesh = TriangleMesh.create_box(width=scale*(x_max - x_min),
                                              height=scale*(y_max - y_min),
                                              depth=d)
        bottom_mesh.translate(-bottom_mesh.get_center())
        bottom_mesh.translate(np.array([model_c[0], model_c[1], z_min]))
        bottom_obj = ('bottom_side', bottom_mesh, sides_material)

        sides_objs_list = [front_obj, right_obj,
                           back_obj, left_obj, top_obj, bottom_obj]

        pcd_obj = self.spcd.get_pcd()
        regions_objs_list = []

        path_objs_list = []

        if not self.selected_side:
            for i, region in enumerate(self.regions):
                selected = i == settings.selected_region
                region_objs = region.get_objs(selected=selected)
                regions_objs_list.append(region_objs)
        else:
            lines = []
            points = []
            for idx, i in enumerate(self.sides[self.selected_side]):
                selected = i == settings.selected_region
                region = self.regions[i]
                region_objs = region.get_objs(selected=selected)
                regions_objs_list.append(region_objs)
                position, _ = region.viewpoint.get_position_orientation()
                if idx < len(self.sides[self.selected_side])-1:
                    lines.append([idx, idx + 1])
                points.append(position)
            if len(points) > 1:
                path_line_set = o3d.geometry.LineSet()
                path_line_set.lines = o3d.utility.Vector2iVector(
                    np.array(lines))
                path_line_set.points = o3d.utility.Vector3dVector(
                    np.array(points))

                path_material = MaterialRecord()
                path_material.shader = 'defaultLit'
                path_material.base_color = [1., 1., 1., 1.]
                path_objs_list.append(('path', path_line_set, path_material))

                # Add start and end to path
                radius = 5
                start_mesh = TriangleMesh.create_octahedron(radius=radius)
                start_mesh.compute_vertex_normals()
                start_mesh.transform(
                    self.regions[self.sides[self.selected_side][0]].viewpoint.tf)
                start_material = MaterialRecord()
                start_material.shader = 'defaultLitTransparency'
                start_material.base_color = [0., 1., 0., 0.5]
                path_objs_list.append(('start', start_mesh, start_material))
                end_mesh = TriangleMesh.create_octahedron(radius=radius)
                end_mesh.compute_vertex_normals()
                end_mesh.transform(
                    self.regions[self.sides[self.selected_side][-1]].viewpoint.tf)
                end_material = MaterialRecord()
                end_material.shader = 'defaultLitTransparency'
                end_material.base_color = [1., 0., 0., 0.5]
                path_objs_list.append(('end', end_mesh, end_material))

        # for region in self.regions:
        #     region_objs = region.get_objs()
        #     regions_objs_list.append(region_objs)
        return model_obj, pcd_obj, regions_objs_list, sides_objs_list, path_objs_list
