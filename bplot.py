import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from io import BytesIO
import base64
import plotly.graph_objects as go

class BPlot:
    def __init__(self, filename):
        self.filename = "alphapose-results-"+filename+".json"
        self.category = 0
        self.jsondata = {}
        self.distances = []
        self.slopes = []
        self.avg = 0

    def convert(self):
        with open(self.filename,'rb') as fh:
            data = json.load(fh)

        mmskeleton_data = {
            "data": {
                "keypoint": []
            },
            "frame_dir": "video_name",  # Set video name here
            "img_shape": [720, 1280],  # Example: height and width of the video
            "original_shape": [720, 1280],
            "total_frames": len(data)
        }


        bounding = []
        for frame_data in data:
            keypoints = frame_data["keypoints"]
            frame_keypoints = []
            # Group keypoints in sets of 3 (x, y, confidence)
            for i in range(0, len(keypoints), 3):
                frame_keypoints.append([keypoints[i], keypoints[i + 1], keypoints[i + 2]])
            # Append to the keypoint list in MMSkeleton format
            mmskeleton_data["data"]["keypoint"].append([frame_keypoints])
            bounding.append(frame_data["box"])

        # Example label for the video, adjust accordingly
        mmskeleton_data["data"]["label"] = 0

        # Convert to JSON
        self.jsondata = json.dumps(mmskeleton_data, indent=4)
        """ print(json_data)
        print(bounding) """
        return self.jsondata

    def plot_nx_graph(self):
        json_data = json.loads(self.jsondata)

        """ skeleton_edges = [
            (47, 48), (48, 49), (49, 50), (50, 51), 
            (47, 52), (52, 53), (53, 54), (54, 55),
            (47, 56), (56, 57), (57, 58), (58, 59),
            (47, 60), (60, 61), (61, 62), (62, 63),
            (47, 64), (64, 65), (65, 66), (66, 67),
        ] """
        skeleton_edges = [
            (0, 1), (18, 17), (18, 19), (18, 25), (19, 20), (20, 21), 
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 17), (1, 2), (2, 3), (3, 4),
        ]
        hand_parts = [47, 64, 65, 66, 67, 60, 61, 62, 63, 56, 57, 58, 59, 52, 53, 54, 55, 48, 49, 50, 51]

        """ skeleton_edges = [
            (15, 27), (27, 28), (28, 29), (29, 30), 
            (15, 31), (31, 32), (32, 33), (33, 34),
            (15, 35), (35, 36), (36, 37), (37, 38),
            (15, 39), (39, 40), (40, 41), (41, 42),
            (15, 43), (43, 44), (44, 45), (45, 46),
        ] """
        frames = json_data["data"]["keypoint"]
        main_kp = []
        # Iterate through frames and plot every 20th frame
        for frame_index in range(len(frames)):
                if frame_index % 10 == 0:
                    frame_keypoints = frames[frame_index][0]  # Get keypoints for the current frame

                    # Extract (x, y) positions, ignoring confidence scores
                    keypoints = [(frame_keypoints[kp][0], frame_keypoints[kp][1]) for kp in range(len(frame_keypoints)) if kp in hand_parts]
                    main_kp.append(keypoints)
                    # Create a graph to plot keypoints and connections
                    G = nx.Graph()
                    for idx, (x, y) in enumerate(keypoints):
                        G.add_node(idx, pos=(x, y))

                    # Add edges based on the skeleton structure
                    for edge in skeleton_edges:
                        if edge[0] < len(keypoints) and edge[1] < len(keypoints):  # Check if keypoints exist
                            G.add_edge(edge[0], edge[1])

                    # Plotting
                    plt.figure(figsize=(8, 8))
                    pos = nx.get_node_attributes(G, 'pos')
                    nx.draw(G, pos, node_size=400, node_color='lightblue', with_labels=True)
                    nx.draw_networkx_edges(G, pos, width=2, edge_color='orange')  # Draw edges
                    plt.title(f"Skeleton Plot for Frame {frame_index}")
                    plt.gca().invert_yaxis()
                    #plt.show()

    def plot_amplitude(self):
        json_data = self.jsondata
        if isinstance(json_data, str):
            json_data = json.loads(json_data)

        self.distances = []
        frames = json_data["data"]["keypoint"]

        def calculate_distance(x1, y1, x2, y2):
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        for frame_index in range(len(frames)):
            frame = frames[frame_index][0]
            x55, y55, c55 = frame[55]  # 55th keypoint (index 54)
            x51, y51, c51 = frame[51]  # 51st keypoint (index 50)
            distance = calculate_distance(x55, y55, x51, y51)
            self.distances.append(distance)

        distances_array = np.array(self.distances)
        self.avg = np.mean(self.distances)

        # Find peaks with prominence 10
        peaks_indices, _ = find_peaks(distances_array, distance=5, prominence=10)
        highest_points = distances_array[peaks_indices]

        fig = go.Figure()

        # Plot peaks first
        fig.add_trace(go.Scatter(x=peaks_indices + 1, y=highest_points, mode='markers', name='Peaks', marker=dict(color='green', size=8)))

        # Interpolation: add one point between every two consecutive peaks
        interp_x = []
        interp_y = []
        for i in range(len(peaks_indices) - 1):
            x_start, x_end = peaks_indices[i] + 1, peaks_indices[i + 1] + 1
            y_start, y_end = highest_points[i], highest_points[i + 1]

            # Midpoint between the peaks
            mid_x = (x_start + x_end) / 2
            mid_y = (y_start + y_end) / 2

            # Collect interpolation points (peak and midpoint)
            interp_x.extend([x_start, mid_x, x_end])
            interp_y.extend([y_start, mid_y, y_end])

        # Plot interpolated data
        fig.add_trace(go.Scatter(x=interp_x, y=interp_y, mode='lines+markers', name='Interpolated Points', line=dict(color='orange')))

        # Define sections and add vertical lines to separate them
        sections = [(0, 150), (150, 300), (300, 450)]
        section_colors = ['blue', 'red', 'green']  # Different colors for each section
        self.slopes = []

        for i, (start, end) in enumerate(sections):
            # Add vertical lines to show section boundaries
            fig.add_vline(x=start, line_width=2, line_dash='dash', line_color='black')
            fig.add_vline(x=end, line_width=2, line_dash='dash', line_color='black')

            # Extract section's peaks
            section_indices = [j for j, x in enumerate(peaks_indices + 1) if start <= x <= end]
            section_x = np.array([peaks_indices[j] + 1 for j in section_indices])
            section_y = np.array([highest_points[j] for j in section_indices])

            # Perform regression for the section
            if len(section_x) > 1:
                coeffs = np.polyfit(section_x, section_y, 1)
                slope = coeffs[0]
                self.slopes.append(slope)

                # Create regression line for the section
                regression_line = np.poly1d(coeffs)
                x_range = np.linspace(start, end, 100)
                fig.add_trace(go.Scatter(x=x_range, y=regression_line(x_range), mode='lines', name=f'Section {i + 1} (Slope: {slope:.2f})', line=dict(color=section_colors[i])))

        # Set y-axis limit to 500
        fig.update_layout(title='Highest Amplitudes Over Time (Peaks and Interpolation)',
                        xaxis_title='Frame',
                        yaxis_title='Distance between the thumb and forefinger',
                        template="plotly_white",
                        yaxis=dict(range=[0, 500]),
                        dragmode='zoom')

        return fig.to_html(full_html=False)


    def plot_speed(self):
        frame_rate = 30
        time_interval = 1 / frame_rate
        speeds = np.abs(np.diff(self.distances)) / time_interval
        time_points = np.arange(1, len(speeds) + 1)

        fig = go.Figure()

        # Add speed data
        fig.add_trace(go.Scatter(x=time_points, y=speeds, mode='markers+lines', name='Speed', marker=dict(color='orange')))

        # Fit a linear regression line
        X = time_points.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, speeds)
        regression_line = model.predict(X)

        # Add regression line
        fig.add_trace(go.Scatter(x=time_points, y=regression_line, mode='lines', name='Trend Line', line=dict(color='blue')))

        fig.update_layout(title='Speed of Movements Between Keypoints Over Time',
                        xaxis_title='Frame',
                        yaxis_title='Speed (units/second)',
                        template="plotly_white",
                        yaxis=dict(range=[0, 8000]),
                        dragmode='zoom')

        return fig.to_html(full_html=False)

    def determine_severity(self):
        a = self.slopes[0]
        b = self.slopes[1]
        c = self.slopes[2]
        if a <= -1:
            self.category = 3
        elif b <= -1:
            self.category = 2
        elif c <= -1:
            self.category = 1
        else:
            print(self.avg)
            if self.avg > 190:
                self.category = 0
            else:
                self.category = 4
        return self.category



