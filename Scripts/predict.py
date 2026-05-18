# -----------------------------------------------------------------------#
# 宋进雨
# 目标检测定位后处理
# -----------------------------------------------------------------------#
import sys
import cv2
import open3d as o3d
from PIL import Image
import numpy as np
from Scripts.yolo import YOLO
from collections import Counter
import pyrealsense2 as rs
import Scripts.share_vars as share_vars


class Location:
    def __init__(self):
        # pose_song
        self.pos_photo = np.array([309.82, 185.90, 696.45, np.deg2rad(179.30), np.deg2rad(-0.41), np.deg2rad(-135.71)])
        # pose_wang
        # self.pos_photo = np.array([309.82, 185.90, 696.45, np.deg2rad(179.30), np.deg2rad(-0.41), np.deg2rad(-135.71)])
        self.pos_eye2arm = np.array([[ 0.03832771,0.99896093,-0.02465866,-85.84213393],
                                     [-0.99922616,0.03853273,0.00789369,31.8694465],
                                     [ 0.00883565,0.02433703,0.99966476,13.28287818],
                                     [0, 0, 0, 1]])
        self.path = 'img/depth_roi/point_cloud_roi.pcd'
        self.grap_log = 'X_Max'
        self.drink_type = share_vars.global_drink_type
        print(f"drink_type:{share_vars.global_drink_type}")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        try:
            cof = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            profile = cof.get_stream(rs.stream.color)
            intrinsics = profile.as_video_stream_profile().get_intrinsics()
            self.fx = intrinsics.fx
            self.fy = intrinsics.fy
            self.cx = intrinsics.ppx
            self.cy = intrinsics.ppy
            print(f"内参",self.fx, self.fy, self.cx, self.cy)
            print('相机已连接')
        except RuntimeError:
            print('检查相机连接!')


    def get_data_from_cam(self):
        global aligned_frames
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
        except RuntimeError:
            print('检查相机连接!')
        try:
            # 获取对齐后的深度帧和彩色帧
            aligned_color_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()
            
            col_im = np.asanyarray(aligned_color_frame.get_data())
            dep_im = np.asanyarray(aligned_depth_frame.get_data())

        except RuntimeError as e:
            print('图像获取失败！')
            print(f'错误详情：{e}')
        except NameError:
            print("相机流获取失败")
        except AttributeError as e:
            print(f'获取内参失败：{e}')

        print("获取数据成功")
        print(f"深度图形状：{dep_im.shape}, RGB 图形状：{col_im.shape}")
        print(f"深度值范围：{dep_im.min()} - {dep_im.max()}")
        return col_im, dep_im


    def save_as_pcd(self, filename, points):
        """
                使用 NumPy 将点云保存为 ASCII 格式的 PCD 文件
                :param filename: 输出文件名（如 "output.pcd"）
                :param points: 点云数据，形状为 (N, 3) 的 NumPy 数组
                """
        header = f"""\
            # .PCD v0.7 - Point Cloud Data file format
            VERSION 0.7
            FIELDS x y z
            SIZE 4 4 4
            TYPE F F F
            COUNT 1 1 1
            WIDTH {len(points)}
            HEIGHT 1
            VIEWPOINT 0 0 0 1 0 0 0
            POINTS {len(points)}
            DATA ascii
            """
        with open(filename, "w") as f:
            f.write(header)
            np.savetxt(f, points, fmt="%.6f")  # 保存点坐标，保留 6 位小数

    def put_img_into_mode(self, image):
        class_name = []
        pixel_conner = np.empty((), np.float32)
        yolo = YOLO()
        crop = False
        count = True
        color_save_path = 'img/result/detect_source_picture.jpg'
        cv2.imwrite(color_save_path, image)
        try:
            img_input = Image.open(color_save_path)
        except Exception as e:
            raise e
        try:
            class_name, pixel_conner = yolo.detect_image(img_input, crop=crop, count=count)  # 不能输入数组格式的图片
            # print(class_name, pixel_conner)
        except TypeError as e:
            print(f'错误:{e}，相机距离目标位置过近，请调整相机距离')
        print('目标检测已完成')
        return class_name, pixel_conner

    def logic_sort_select_one(self, class_name, pixel_conner, dep_image, drink_type):
        num_roi = 0
        list_pos_x = []
        list_pos_y = []
        list_pos_z = []
        j = 0
        k = 0
        try:
            for i in range(len(class_name)):
                if class_name[i] == drink_type:
                    num_roi = num_roi + 1
                    j = i
                    left = pixel_conner[i, 0]
                    right = pixel_conner[i, 1]
                    top = pixel_conner[i, 2]
                    bottom = pixel_conner[i, 3]
                    # 增加抓取逻辑，选取场景中最近位置的饮品
                    center_x_pixel = round((left + right) / 2)
                    center_y_pixel = round((top + bottom) / 2)
                    dep_cen = dep_image[center_y_pixel, center_x_pixel]
                    pos_x_in_cam = (center_x_pixel - self.cx) / self.fx * dep_cen
                    pos_y_in_cam = (center_y_pixel - self.cy) / self.fy * dep_cen
                    pos_c = np.array([pos_x_in_cam, pos_y_in_cam, dep_cen, 1])
                    pos_eb = self.tans_eb2mix(self.pos_photo[0], self.pos_photo[1], self.pos_photo[2],
                                              self.pos_photo[3], self.pos_photo[4], self.pos_photo[5])
                    poh_ce = self.pos_eye2arm
                    pos_in_rob = pos_eb @ poh_ce @ pos_c.T
                    list_pos_x.append(pos_in_rob[0])
                    list_pos_y.append(pos_in_rob[1])
                    list_pos_z.append(pos_in_rob[2])
                else:
                    continue
            if self.grap_log == 'X_Max':
                indices = np.argsort(list_pos_x)  # 注：排序后返回的不是列表值是列表索引
                try:
                    k = indices[0]
                except IndexError:
                    print(f'制定抓取逻辑时，检测到可供抓取的{drink_type}数量为0')
            elif self.grap_log == 'X_Min':
                indices = np.sort(list_pos_x)
                try:
                    k = indices[0]
                except IndexError:
                    print(f'制定抓取逻辑时，检测到可供抓取的{drink_type}数量为0')
            elif self.grap_log == 'Y_Max':
                indices = np.argsort(list_pos_y)
                try:
                    k = indices[0]
                except IndexError:
                    print(f'制定抓取逻辑时，检测到可供抓取的{drink_type}数量为0')
            elif self.grap_log == 'Y_Min':
                indices = np.sort(list_pos_y)
                try:
                    k = indices[0]
                except IndexError:
                    print(f'制定抓取逻辑时，检测到可供抓取的{drink_type}数量为0')
            elif self.grap_log == 'Z_Max':
                indices = np.argsort(list_pos_z)
                try:
                    k = indices[0]
                except IndexError:
                    print(f'制定抓取逻辑时，检测到可供抓取的{drink_type}数量为0')
            elif self.grap_log == 'Z_Min':
                indices = np.sort(list_pos_z)
                try:
                    k = indices[0]
                except IndexError:
                    print(f'制定抓取逻辑时，检测到可供抓取的{drink_type}数量为0')
            else:
                print('请确认抓取逻辑设置是否正确')
        except UnboundLocalError as e:
            print(f"错误:{e}，相机距离目标太近，导致无法返回目标类别，请调整")
            sys.exit()  # 退出
        if num_roi == 0:
            print(f'场景中未检测到有饮品{drink_type}')
        else:
            print(f'场景中有{num_roi}瓶{drink_type}，将为你挑选距离你最近的一瓶')
        return num_roi, j, k

    def trans_png_roi_to_point_cloud(self, pixel_conner, dep_image, num_roi, j, k):
        target_index = j - num_roi + k + 1  if num_roi > 0 else 0

        left = pixel_conner[target_index, 0]
        right = pixel_conner[target_index, 1]
        top = pixel_conner[target_index, 2]
        bottom = pixel_conner[target_index, 3]

        left = max(0, int(left))
        right = min(dep_image.shape[1], int(right))
        top = max(0, int(top))
        bottom = min(dep_image.shape[0], int(bottom))

        # 添加边距扩展，避免边缘深度噪声
        margin = 0
        left_expanded = max(0, left - margin)
        right_expanded = min(dep_image.shape[1], right + margin)
        top_expanded = max(0, top - margin)
        bottom_expanded = min(dep_image.shape[0], bottom + margin)
        dep_roi = dep_image[top_expanded:bottom_expanded,left_expanded:right_expanded]
        roi_height, roi_width = dep_roi.shape

        mesh_u, mesh_v = np.meshgrid(np.arange(roi_width),np.arange(roi_height))

        point_location_x = (mesh_u + left_expanded - self.cx) / self.fx
        point_location_y = (mesh_v + top_expanded - self.cy) / self.fy
        point_cloud = np.zeros((roi_height, roi_width, 3),np.float32)
        point_cloud[:, :, 0] = point_location_x * dep_roi
        point_cloud[:, :, 1] = point_location_y * dep_roi
        point_cloud[:, :, 2] = dep_roi
        points = np.reshape(point_cloud, (-1, 3))
        valid_points2 = points[points[:, 2] > 0]
        try:
            self.save_as_pcd(f"img/depth_roi/point_cloud_roi.pcd", valid_points2)
            print(f"第{num_roi}部分点云PCD文件已保存成功!")
        except RuntimeError as e:
            print(f"PCD文件{num_roi}保存失败!", e)
        dep_save_path = f"img/depth_roi/dep_roi.png"
        dep_roi_normalized = cv2.normalize(dep_roi, None, 0, 65535, cv2.NORM_MINMAX)
        cv2.imwrite(dep_save_path, dep_roi_normalized.astype(np.uint16))

    def tans_matrix2list(self, base_end):
        R_list = np.array([1, 1, 1], np.float32)
        belt = np.atan2(-base_end[2, 0], np.sqrt(base_end[0, 0] ** 2 + base_end[1, 0] ** 2))
        alpha = np.atan2(base_end[1, 0] / np.cos(belt), base_end[0, 0] / np.cos(belt))
        theta = np.atan2(base_end[2, 1] / np.cos(belt), base_end[2, 2] / np.cos(belt))
        alpha_deg = np.rad2deg(alpha)
        belt_deg = np.rad2deg(belt)
        theta_deg = np.rad2deg(theta)
        R_list[0] = theta_deg
        R_list[1] = belt_deg
        R_list[2] = alpha_deg
        return R_list

    def point_cloud_process(self, path):
        pos_target = []
        input_pcd = o3d.io.read_point_cloud(path)
        pcd_filter = input_pcd
        pcd_filter_z = pcd_filter.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=(-np.inf, -np.inf, 400),
                                                                           max_bound=(np.inf, np.inf, 500)))
        print(f"点云滤波后点数:{len(pcd_filter_z.points)}")
        nk = 5
        std_rio = 0.5
        sor_pcd, index = pcd_filter_z.remove_statistical_outlier(nk, std_rio)
        points_num = len(sor_pcd.points)
        if points_num == 0:
            return pos_target
        else:
            eps = 20
            min_points = 50
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                labels = np.array(sor_pcd.cluster_dbscan(eps, min_points, print_progress=True))
            valid_labels = labels[labels != -1]
            label_counts = Counter(valid_labels)
            if not label_counts:
                print("警告:未找到任何有效聚类")
            largest_label = max(label_counts, key=label_counts.get)
            largest_indices = np.where(labels == largest_label)[0]
            largest_cluster = sor_pcd.select_by_index(largest_indices)
            dis_threshold = 4
            ransac_n = 8
            num_iterations = 1000
            plane_mode, inlines = largest_cluster.segment_plane(dis_threshold, ransac_n, num_iterations)
            plane_in_could = largest_cluster.select_by_index(inlines)
            A = plane_mode[0]
            B = plane_mode[1]
            C = plane_mode[2]
            D = plane_mode[3]
            Xcoff = np.array([B * B + C * C, -A * B, -A * C])
            Ycoff = np.array([-B * A, A * A + C * C, -B * C])
            Zcoff = np.array([-A * C, -B * C, A * A + B * B])
            points = np.asarray(plane_in_could.points)
            xp = np.dot(points, Xcoff) - A * D
            yp = np.dot(points, Ycoff) - B * D
            zp = np.dot(points, Zcoff) - C * D
            project_points = np.c_[xp, yp, zp]
            project_cloud = o3d.geometry.PointCloud()
            project_cloud.points = o3d.utility.Vector3dVector(project_points)
            plane_in_could_array = np.asarray(project_cloud.points)
            if len(plane_in_could_array)==0:
                print("未分割出有效点云")
                return None
            self.save_as_pcd(f"img/depth_roi/p1.pcd", plane_in_could_array) # 拟合
            centroid = [np.mean(plane_in_could_array[:, 0]), np.mean(plane_in_could_array[:, 1]),
                        np.mean(plane_in_could_array[:, 2])]
            P_oc = np.array([0,0,0,1], np.float64)
            T_ob = np.eye(4)
            Rx_ob = np.array([[1, 0, 0],
                              [0, np.cos(np.pi), -np.sin(np.pi)],
                              [0, np.sin(np.pi), np.cos(np.pi)]])
            T_eb = self.tans_eb2mix(self.pos_photo[0], self.pos_photo[1], self.pos_photo[2],
                                    self.pos_photo[3], self.pos_photo[4], self.pos_photo[5])
            T_ce = self.pos_eye2arm
            P_oc[:3] = centroid
            t_ob = T_eb @ T_ce @ P_oc.T

            T_ob[:3, :3] = Rx_ob
            T_ob[:3, 3] = t_ob[:3]
            r_list = self.tans_matrix2list(T_ob)
            pos_target.append(np.float16(t_ob[0]))
            pos_target.append(np.float16(t_ob[1]))
            pos_target.append(np.float16(t_ob[2]))
            pos_target.append(np.float16(r_list[0]))
            pos_target.append(np.float16(r_list[1]))
            pos_target.append(np.float16(r_list[2]))
            # print(f'位置结果{pos_target}')
            return pos_target

    def tans_eb2mix(self, x, y, z, a, b, c):
        arm_eb = np.eye(4)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(a), -np.sin(a)],
                       [0, np.sin(a), np.cos(a)]])
        Ry = np.array([[np.cos(b), 0, np.sin(b)],
                       [0, 1, 0],
                       [-np.sin(b), 0, np.cos(b)]])
        Rz = np.array([[np.cos(c), -np.sin(c), 0],
                       [np.sin(c), np.cos(c), 0],
                       [0, 0, 1]])
        R = Rz @ Ry @ Rx
        arm_eb[:3, :3] = R
        arm_eb[0, 3] = x
        arm_eb[1, 3] = y
        arm_eb[2, 3] = z
        return arm_eb

    def trans_data_to_socket(self, pos_list):
        socket_list = f''
        try:
            socket_list = f'ok,{pos_list[0]},{pos_list[1]},{pos_list[2]},{pos_list[3]},{pos_list[4]},{pos_list[5]}'
        except IndexError:
            print(f'{socket_list}超出索引')
        return socket_list

    def img2pos(self, drink_type):

        a, b = self.get_data_from_cam()
        c, d = self.put_img_into_mode(a)
        e, f, g = self.logic_sort_select_one(c, d, b, drink_type)
        self.trans_png_roi_to_point_cloud(d, b, e, f, g)
        h = self.point_cloud_process(self.path)
        # print(h)
        return h

if __name__ == '__main__':
    la = Location()
    h = la.img2pos(la.drink_type)
    print(h)

