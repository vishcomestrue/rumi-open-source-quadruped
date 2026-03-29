import viser
import numpy as np
import plotly.graph_objects as go
from collections import deque
import math
import time


class IMUViewer:
    def __init__(self, port=8080, buffer_size=250):
        print(f"Starting Viser server on port {port}...")
        self.server = viser.ViserServer(port=port)
        self.buffer_size = buffer_size
        self.start_time = None
        self._last_plot_time = 0.0

        self.time_buffer = deque(maxlen=buffer_size)
        self.gyro_buffer = [deque(maxlen=buffer_size) for _ in range(3)]
        self.accel_buffer = [deque(maxlen=buffer_size) for _ in range(3)]
        self.euler_buffer = [deque(maxlen=buffer_size) for _ in range(3)]

        self._setup_3d_scene()
        self._setup_plots()
        print(f"✓ Visualization ready at http://localhost:{port}")

    def _setup_3d_scene(self):
        self.server.scene.add_frame(
            name="/world",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            show_axes=True,
            axes_length=0.045,
            axes_radius=0.002,
        )
        self.imu_frame = self.server.scene.add_frame(
            name="/world/imu",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            show_axes=True,
            axes_length=0.03,
            axes_radius=0.001,
        )
        self.server.scene.add_box(
            name="/world/imu/sensor",
            dimensions=(0.03, 0.025, 0.002),
            color=(20, 20, 20),
            position=(0.0, 0.0, 0.0),
        )
        self.server.scene.add_box(
            name="/world/imu/led",
            dimensions=(0.001, 0.002, 0.001),
            color=(139, 0, 0),
            position=(0.013, 0.005, 0.0015),
        )
        self.server.scene.add_grid(
            name="/ground",
            width=0.3, height=0.3,
            cell_size=0.05, cell_thickness=1.0,
            position=(0.0, 0.0, -0.02),
        )

    def _setup_plots(self):
        def make_plot(title, labels, colors):
            fig = go.Figure()
            for label, color in zip(labels, colors):
                fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name=label,
                                         line=dict(color=color, width=2)))
            fig.update_layout(title=title, xaxis_title="Time (s)", height=250,
                               margin=dict(l=40, r=40, t=40, b=40), showlegend=True)
            return fig

        self.gyro_fig = make_plot("Gyroscope (rad/s)", ["X", "Y", "Z"], ["red", "green", "blue"])
        self.accel_fig = make_plot("Accelerometer (m/s²)", ["X", "Y", "Z"], ["red", "green", "blue"])
        self.euler_fig = make_plot("Euler (deg)", ["Roll", "Pitch", "Yaw"], ["orange", "purple", "cyan"])

        self.gyro_plotly = self.server.gui.add_plotly(figure=self.gyro_fig)
        self.accel_plotly = self.server.gui.add_plotly(figure=self.accel_fig)
        self.euler_plotly = self.server.gui.add_plotly(figure=self.euler_fig)

    @staticmethod
    def _quat_to_euler(qx, qy, qz, qw):
        roll = math.degrees(math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
        sinp = 2*(qw*qy - qz*qx)
        pitch = math.degrees(math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp))
        yaw = math.degrees(math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
        return roll, pitch, yaw

    def update(self, data, position=None):
        if self.start_time is None:
            self.start_time = data['timestamp']
        self.time_buffer.append(data['timestamp'] - self.start_time)

        gx, gy, gz = data['gyro']
        ax, ay, az = data['accel']
        qx, qy, qz, qw = data['quaternion']

        self.gyro_buffer[0].append(gx)
        self.gyro_buffer[1].append(gy)
        self.gyro_buffer[2].append(gz)
        self.accel_buffer[0].append(ax)
        self.accel_buffer[1].append(ay)
        self.accel_buffer[2].append(az)

        roll, pitch, yaw = self._quat_to_euler(qx, qy, qz, qw)
        self.euler_buffer[0].append(roll)
        self.euler_buffer[1].append(pitch)
        self.euler_buffer[2].append(yaw)

        self.imu_frame.wxyz = np.array([qw, qx, qy, qz], dtype=np.float64)
        if position is not None:
            self.imu_frame.position = np.array(position, dtype=np.float64)

        now = time.time()
        if now - self._last_plot_time >= 0.1:
            self._last_plot_time = now
            t_list = list(self.time_buffer)
            for fig, widget, bufs in [
                (self.gyro_fig,  self.gyro_plotly,  self.gyro_buffer),
                (self.accel_fig, self.accel_plotly, self.accel_buffer),
                (self.euler_fig, self.euler_plotly, self.euler_buffer),
            ]:
                for i, buf in enumerate(bufs):
                    fig.data[i].x = t_list
                    fig.data[i].y = list(buf)
                widget.figure = fig


if __name__ == "__main__":
    viewer = IMUViewer(port=8080)
    print("Open http://localhost:8080")

    for i in range(500):
        t = i * 0.02
        sy, cy = math.sin(t * 0.5), math.cos(t * 0.5)
        viewer.update({
            'timestamp': time.time(),
            'quaternion': (0.0, 0.0, sy, cy),   # rotation around Z, (x,y,z,w)
            'accel': (math.sin(t), math.cos(t), 9.8),
            'gyro': (0.1 * math.sin(t), 0.2 * math.cos(t), 0.5),
        })
        time.sleep(0.02)
