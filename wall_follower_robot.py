import time
import math
import os
import csv
import json
from datetime import datetime
from robomaster import robot
import threading

wall_threshold = 500  # mm
yaw_angles = [0, 90, -90, 180]  # หน้า, ขวา, ซ้าย, หลัง


class WallFollowerRobot:
    def __init__(self):
        # sensing
        self.distances = {}
        self.data_ready_event = threading.Event()
        self.current_scan_yaw = None

        # robot handles
        self.ep_robot = None
        self.ep_sensor = None
        self.ep_gimbal = None
        self.ep_chassis = None

        # pose and map for DFS
        self.center_yaw = 0  # มุมปัจจุบันของหุ่น (0=หันไปข้างหน้าเริ่มต้น), ค่ามุมบวก=หมุนตามเข็มนาฬิกา
        self.pose_x = 0.0    # เมตร (แกน x = ทิศทางด้านหน้าเริ่มต้น)
        self.pose_y = 0.0    # เมตร (แกน y = ด้านขวาเป็นค่าลบ, ด้านซ้ายเป็นค่าบวก จากคอนเวนชันคณิตศาสตร์)
        self.grid_cell_size = 0.6  # เมตรต่อ 1 cell สำหรับ DFS (ตรงกับระยะก้าวเดินหลัก)
        self.position_history = [(0.0, 0.0)]  # เก็บพิกัดเมตริกตามลำดับการเดิน
        self.grid_history = [(0, 0)]          # เก็บพิกัดกริดตามลำดับการเดิน
        self.visited_cells = set([(0, 0)])    # สำหรับ DFS

        # logging config
        self.log_dir = "/workspace/logs"
        self.metric_csv_path = os.path.join(self.log_dir, "positions_metric.csv")
        self.grid_csv_path = os.path.join(self.log_dir, "grid_cells.csv")
        self.mapping_json_path = os.path.join(self.log_dir, "mapping.json")

        self.lock = threading.Lock()

    # ----------------------------
    # Logging helpers
    # ----------------------------
    def _ensure_log_dir(self):
        os.makedirs(self.log_dir, exist_ok=True)

    def _init_logs_if_needed(self):
        self._ensure_log_dir()
        # metric CSV header
        if not os.path.exists(self.metric_csv_path):
            with open(self.metric_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp_iso", "step_index", "x_m", "y_m", "yaw_deg", "action"]) 
        # grid CSV header
        if not os.path.exists(self.grid_csv_path):
            with open(self.grid_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp_iso", "step_index", "cell_x", "cell_y", "action"]) 
        # initial JSON mapping file
        if not os.path.exists(self.mapping_json_path):
            self._write_mapping_json(action="init")

    def _write_mapping_json(self, action="update"):
        now_iso = datetime.utcnow().isoformat() + "Z"
        # convert tuples to lists for JSON
        pos_hist = [[float(x), float(y)] for x, y in self.position_history]
        grid_hist = [[int(cx), int(cy)] for cx, cy in self.grid_history]
        visited = [[int(cx), int(cy)] for (cx, cy) in sorted(self.visited_cells)]
        cell_x = int(round(self.pose_x / self.grid_cell_size))
        cell_y = int(round(self.pose_y / self.grid_cell_size))
        data = {
            "last_updated": now_iso,
            "action": action,
            "pose": {"x_m": self.pose_x, "y_m": self.pose_y, "yaw_deg": self.center_yaw},
            "cell": {"cell_x": cell_x, "cell_y": cell_y, "cell_size_m": self.grid_cell_size},
            "position_history": pos_hist,
            "grid_history": grid_hist,
            "visited_cells": visited,
        }
        with open(self.mapping_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _log_metric_row(self, action: str):
        now_iso = datetime.utcnow().isoformat() + "Z"
        step_idx = len(self.position_history) - 1
        with open(self.metric_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([now_iso, step_idx, f"{self.pose_x:.6f}", f"{self.pose_y:.6f}", int(self.center_yaw), action])

    def _log_grid_row(self, action: str, cell_x: int, cell_y: int):
        now_iso = datetime.utcnow().isoformat() + "Z"
        step_idx = len(self.grid_history) - 1
        with open(self.grid_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([now_iso, step_idx, int(cell_x), int(cell_y), action])

    # ----------------------------
    # Utilities for pose handling
    # ----------------------------
    def _normalize_360(self, deg):
        v = deg % 360
        return v if v >= 0 else v + 360

    def _normalize_180(self, deg):
        v = (deg + 180) % 360 - 180
        return v

    def _update_pose_by_forward(self, distance_m: float, action: str = "move"):
        # แปลงมุมภายใน (บวกตามเข็ม) ให้เป็นมุมคณิตศาสตร์ (บวกทวนเข็ม) เพื่อคำนวณ cos/sin
        theta = -math.radians(self.center_yaw)
        dx = distance_m * math.cos(theta)
        dy = distance_m * math.sin(theta)

        self.pose_x += dx
        self.pose_y += dy

        # อัปเดตพิกัดกริด (ปัดเป็น cell)
        cell_x = int(round(self.pose_x / self.grid_cell_size))
        cell_y = int(round(self.pose_y / self.grid_cell_size))

        # เก็บ history (เฉพาะเมื่อมีการเปลี่ยนตำแหน่งจริง)
        last_metric = self.position_history[-1]
        if abs(self.pose_x - last_metric[0]) > 1e-6 or abs(self.pose_y - last_metric[1]) > 1e-6:
            self.position_history.append((self.pose_x, self.pose_y))
            # log metric step
            self._log_metric_row(action)

        last_cell = self.grid_history[-1]
        cell_changed = False
        if (cell_x, cell_y) != last_cell:
            self.grid_history.append((cell_x, cell_y))
            self.visited_cells.add((cell_x, cell_y))
            cell_changed = True
            # log grid step
            self._log_grid_row(action, cell_x, cell_y)

        # update JSON snapshot
        self._write_mapping_json(action=action)

        print(f"[POSE] x={self.pose_x:.3f} m, y={self.pose_y:.3f} m, yaw={self.center_yaw}°, cell=({cell_x},{cell_y})")
        return cell_changed

    # ----------------------------
    # Sensors
    # ----------------------------
    def sub_distance_handler(self, distance):
        tof1 = distance[0]
        with self.lock:
            if self.current_scan_yaw is not None:
                self.distances[self.current_scan_yaw] = tof1
                print(f"[DEBUG] Yaw={self.current_scan_yaw}°, Distance={tof1} mm")
                if len(self.distances) == len(yaw_angles):
                    self.data_ready_event.set()

    # ----------------------------
    # Motion
    # ----------------------------
    def turn_to_yaw(self, target_yaw, action: str = "turn"):
        # หมุนไปยัง "มุมเป้าหมายแบบสัมบูรณ์" โดยคำนวณมุมหมุนสัมพัทธ์จากมุมปัจจุบัน
        target_yaw = self._normalize_360(target_yaw)
        delta = self._normalize_180(target_yaw - self.center_yaw)
        print(f"[DEBUG] หมุนหุ่นยนต์จาก {self.center_yaw}° ไป {target_yaw}° (หมุน {delta:+.1f}°)")
        if abs(delta) > 1e-3:
            self.ep_chassis.move(x=0, y=0, z=delta, z_speed=45).wait_for_completed()
        self.center_yaw = target_yaw
        # snapshot after turn
        self._write_mapping_json(action=action)

    def scan_all_directions(self):
        with self.lock:
            self.distances.clear()
        self.data_ready_event.clear()

        for yaw in yaw_angles:
            self.current_scan_yaw = yaw
            print(f"[DEBUG] หมุนกิมบอลไปที่ yaw: {yaw} องศา")
            self.ep_gimbal.moveto(pitch=0, yaw=yaw, yaw_speed=250).wait_for_completed()
            time.sleep(0.5)

        if not self.data_ready_event.wait(timeout=3):
            print("[WARNING] ไม่ได้รับข้อมูลครบทุกมุมในเวลาที่กำหนด")

        with self.lock:
            distances_copy = self.distances.copy()
        return distances_copy

    def is_clear(self, dist):
        if dist is None or dist <= 0:
            print(f"[DEBUG] ระยะ {dist} mm ไม่ valid")
            return False
        clear = dist > wall_threshold
        print(f"[DEBUG] ระยะ {dist} mm {'ว่าง' if clear else 'ติดกำแพง'}")
        return clear

    # ----------------------------
    # Main loop
    # ----------------------------
    def run(self):
        # init logs
        self._init_logs_if_needed()

        self.ep_robot = robot.Robot()
        self.ep_robot.initialize(conn_type="ap")

        self.ep_sensor = self.ep_robot.sensor
        self.ep_gimbal = self.ep_robot.gimbal
        self.ep_chassis = self.ep_robot.chassis

        self.ep_sensor.sub_distance(freq=5, callback=self.sub_distance_handler)

        # log initial pose/cell and snapshot mapping
        self._log_metric_row("start")
        init_cell_x = int(round(self.pose_x / self.grid_cell_size))
        init_cell_y = int(round(self.pose_y / self.grid_cell_size))
        self._log_grid_row("start", init_cell_x, init_cell_y)
        self._write_mapping_json(action="start")

        try:
            while True:
                distances = self.scan_all_directions()
                print(f"[INFO] ระยะที่สแกนได้: {distances}")

                front_dist = distances.get(0)
                right_dist = distances.get(90)
                left_dist = distances.get(-90)
                back_dist = distances.get(180)

                print(f"[INFO] front={front_dist}, right={right_dist}, left={left_dist}, back={back_dist}")

                if self.is_clear(right_dist):
                    print("[ACTION] ทางขวาว่าง เลี้ยวขวาแล้วเดิน")
                    self.turn_to_yaw(90, action="turn_right")
                    self.ep_chassis.move(x=0.6, y=0, z=0, xy_speed=0.7).wait_for_completed()
                    self._update_pose_by_forward(0.6, action="move_right")

                elif self.is_clear(front_dist):
                    print("[ACTION] ทางหน้าว่าง เดินตรงไป")
                    self.turn_to_yaw(0, action="turn_front")
                    self.ep_chassis.move(x=0.6, y=0, z=0, xy_speed=0.7).wait_for_completed()
                    self._update_pose_by_forward(0.6, action="move_front")

                elif self.is_clear(left_dist):
                    print("[ACTION] ทางซ้าว่าง เลี้ยวซ้ายแล้วเดิน")
                    self.turn_to_yaw(-90, action="turn_left")
                    self.ep_chassis.move(x=0.6, y=0, z=0, xy_speed=0.7).wait_for_completed()
                    self._update_pose_by_forward(0.6, action="move_left")

                elif self.is_clear(back_dist):
                    print("[ACTION] ทางหลังว่าง หมุนกลับหลังแล้วเดิน")
                    self.turn_to_yaw(180, action="turn_back")
                    self.ep_chassis.move(x=0.6, y=0, z=0, xy_speed=0.7).wait_for_completed()
                    self._update_pose_by_forward(0.6, action="move_back")

                else:
                    print("[ACTION] ทางตันทุกด้าน ถอยหลังเล็กน้อยแล้วหมุนค้นหาใหม่")
                    self.ep_chassis.move(x=-0.3, y=0, z=0, xy_speed=0.5).wait_for_completed()
                    self._update_pose_by_forward(-0.3, action="backtrack")
                    new_yaw = self._normalize_360(self.center_yaw + 90)
                    self.turn_to_yaw(new_yaw, action="search_rotate")

                self.ep_chassis.drive_speed(x=0, y=0, z=0)

        except KeyboardInterrupt:
            print("[INFO] หยุดโปรแกรมด้วยผู้ใช้")
            # แสดงสรุป path ที่เก็บ
            print(f"[RESULT] metric path len={len(self.position_history)}")
            print(f"[RESULT] grid path len={len(self.grid_history)} last={self.grid_history[-1]}")
            print(f"[RESULT] visited cells count={len(self.visited_cells)}")
            # final snapshot
            self._write_mapping_json(action="stopped_by_user")

        self.ep_sensor.unsub_distance()
        self.ep_robot.close()


if __name__ == "__main__":
    robot_controller = WallFollowerRobot()
    robot_controller.run()