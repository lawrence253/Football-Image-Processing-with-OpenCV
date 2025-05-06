import cv2
import numpy as np
import sys
import os
from ultralytics import YOLO
from sklearn.cluster import KMeans
import pandas as pd
from collections import defaultdict

def get_grass_color(img):
    """
    Finds the color of the grass in the background of the image

    Args:
        img: np.array object of shape (WxHx3) that represents the BGR value of the
        frame pixels .

    Returns:
        grass_color
            Tuple of the BGR value of the grass color in the image
    """
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Calculate the mean value of the pixels that are not masked
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    grass_color = cv2.mean(img, mask=mask)
    return grass_color[:3]

def get_players_boxes(result):
    """
    Finds the images of the players in the frame and their bounding boxes.

    Args:
        result: ultralytics.engine.results.Results object that contains all the
        result of running the object detection algroithm on the frame

    Returns:
        players_imgs
            List of np.array objects that contain the BGR values of the cropped
            parts of the image that contains players.
        players_boxes
            List of ultralytics.engine.results.Boxes objects that contain various
            information about the bounding boxes of the players found in the image.
    """
    players_imgs = []
    players_boxes = []
    for box in result.boxes:
        label = int(box.cls.numpy()[0])
        if label == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
            player_img = result.orig_img[y1: y2, x1: x2]
            players_imgs.append(player_img)
            players_boxes.append(box)
    return players_imgs, players_boxes

def get_kits_colors(players, grass_hsv=None, frame=None):
    """
    Finds the kit colors of all the players in the current frame

    Args:
        players: List of np.array objects that contain the BGR values of the image
        portions that contain players.
        grass_hsv: tuple that contain the HSV color value of the grass color of
        the image background.

    Returns:
        kits_colors
            List of np arrays that contain the BGR values of the kits color of all
            the players in the current frame
    """
    kits_colors = []
    if grass_hsv is None:
        grass_color = get_grass_color(frame)
        grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

    for player_img in players:
        # Chuyển sang không gian màu HSV để xử lý tốt hơn với ánh sáng
        hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)

        # Tạo mask cho màu cỏ với ngưỡng rộng hơn
        lower_green = np.array([grass_hsv[0, 0, 0] - 15, 30, 30])
        upper_green = np.array([grass_hsv[0, 0, 0] + 15, 255, 255])
        grass_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Tạo mask cho vùng áo (1/3 trên của người)
        height = player_img.shape[0]
        upper_third_mask = np.zeros(player_img.shape[:2], np.uint8)
        upper_third_mask[0:height//3] = 255
        
        # Kết hợp các mask
        combined_mask = cv2.bitwise_and(cv2.bitwise_not(grass_mask), upper_third_mask)
        
        # Áp dụng Gaussian blur để giảm nhiễu
        blurred = cv2.GaussianBlur(player_img, (5, 5), 0)
        
        # Tính màu trung bình trong vùng mask
        mean_color = cv2.mean(blurred, mask=combined_mask)
        kit_color = np.array(mean_color[:3])
        
        if not np.any(np.isnan(kit_color)):  # Kiểm tra giá trị hợp lệ
            kits_colors.append(kit_color)
            
    return kits_colors

def get_kits_classifier(kits_colors):
    """
    Creates a K-Means classifier that can classify the kits accroding to their BGR
    values into 2 different clusters each of them represents one of the teams

    Args:
        kits_colors: List of np.array objects that contain the BGR values of
        the colors of the kits of the players found in the current frame.

    Returns:
        kits_kmeans
            sklearn.cluster.KMeans object that can classify the players kits into
            2 teams according to their color..
    """
    kits_kmeans = KMeans(n_clusters=2)
    kits_kmeans.fit(kits_colors);
    return kits_kmeans

def classify_kits(kits_classifer, kits_colors):
    """
    Classifies the player into one of the two teams according to the player's kit
    color

    Args:
        kits_classifer: sklearn.cluster.KMeans object that can classify the
        players kits into 2 teams according to their color.
        kits_colors: List of np.array objects that contain the BGR values of
        the colors of the kits of the players found in the current frame.

    Returns:
        team
            np.array object containing a single integer that carries the player's
            team number (0 or 1)
    """
    team = kits_classifer.predict(kits_colors)
    return team

def get_left_team_label(players_boxes, kits_colors, kits_clf, frame_width):
    """
    Finds the label of the team that is on the left of the screen using multiple frames

    Args:
        players_boxes: List of ultralytics.engine.results.Boxes objects that
        contain various information about the bounding boxes of the players found
        in the image.
        kits_colors: List of np.array objects that contain the BGR values of
        the colors of the kits of the players found in the current frame.
        kits_clf: sklearn.cluster.KMeans object that can classify the players kits
        into 2 teams according to their color.
    Returns:
        left_team_label
            Int that holds the number of the team that's on the left of the image
            either (0 or 1)
    """
    left_team_label = 0
    team_0_left = []  # Số cầu thủ team 0 ở nửa trái
    team_1_left = []  # Số cầu thủ team 1 ở nửa trái
    team_0_positions = []
    team_1_positions = []

    # Phân loại vị trí của từng cầu thủ
    for i in range(len(players_boxes)):
        x1, y1, x2, y2 = map(int, players_boxes[i].xyxy[0].numpy())
        center_x = (x1 + x2) / 2
        
        team = classify_kits(kits_clf, [kits_colors[i]]).item()
        
        if team == 0:
            team_0_positions.append(center_x)
            if center_x < frame_width / 2:
                team_0_left.append(1)
        else:
            team_1_positions.append(center_x)
            if center_x < frame_width / 2:
                team_1_left.append(1)

    # Tính tỉ lệ cầu thủ ở bên trái của mỗi đội
    team_0_left_ratio = len(team_0_left) / len(team_0_positions) if team_0_positions else 0
    team_1_left_ratio = len(team_1_left) / len(team_1_positions) if team_1_positions else 0

    # Nếu một đội có hơn 60% số cầu thủ ở bên trái, đó là đội bên trái
    if team_0_left_ratio > 0.6:
        left_team_label = 0
    elif team_1_left_ratio > 0.6:
        left_team_label = 1
    else:
        # Nếu không rõ ràng, sử dụng vị trí trung bình
        team_0_avg = np.mean(team_0_positions) if team_0_positions else 0
        team_1_avg = np.mean(team_1_positions) if team_1_positions else 0
        left_team_label = 1 if team_0_avg > team_1_avg else 0

    return left_team_label


def interpolate_ball_positions(ball_detections):
    """
    Interpolates missing ball positions using pandas interpolation with improved smoothing
    
    Args:
        ball_detections: List of dictionaries containing frame index and ball coordinates
                        Each dict should have 'frame_idx' and coordinates (x1,y1,x2,y2)
    
    Returns:
        DataFrame with interpolated ball positions
    """
    if not ball_detections:
        return pd.DataFrame()
        
    # Create DataFrame from detections
    df = pd.DataFrame(ball_detections)
    df = df.set_index('frame_idx')
    
    # Sort by frame index to ensure proper interpolation
    df = df.sort_index()
    
    # Apply rolling mean to smooth the trajectories (window size of 3)
    df_smooth = df.rolling(window=3, min_periods=1, center=True).mean()
    
    # Interpolate missing values using cubic interpolation for smoother curves
    df_interp = df_smooth.interpolate(method='cubic', limit_direction='both', limit=30)
    
    # Fill any remaining gaps using forward fill then backward fill
    df_final = df_interp.ffill().bfill()
    
    return df_final

def annotate_video(video_path, model):
    """
    Process video with real-time tracking and annotation
    Args:
        video_path: Path to input video
        model: YOLOv8 model instance
    """
    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup output video
    video_name = video_path.split('\\')[-1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_out.mp4")
    output_video = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    # Initialize variables
    kits_clf = None
    left_team_label = None
    grass_hsv = None
    team_colors = {0: None, 1: None}
    player_tracks = defaultdict(lambda: {"team": None, "color": None, "missed_frames": 0})
    ball_detections = []
    last_ball_pos = None
    
    # Colors for other objects
    other_colors = {
        "4": (155, 62, 157),  # Ball - Purple
        "5": (123, 174, 213), # Main Ref - Light Blue
        "6": (217, 89, 204),  # Side Ref - Pink
        "7": (22, 11, 15)     # Staff - Dark Gray
    }

    print("Processing video...")
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        annotated_frame = cv2.resize(frame, (width, height))

        # Run detection and tracking with ByteTrack
        results = model.track(
            annotated_frame,
            conf=0.3,  # Lowered confidence threshold from 0.5 to 0.3
            verbose=False,
            tracker="bytetrack.yaml",
            persist=True
        )
        
        if not results:
            continue
            
        result = results[0]

        # Process first frame to get team colors
        if frame_count == 1:
            players_imgs, players_boxes = get_players_boxes(result)
            if players_imgs:  # Check if any players were detected
                kits_colors = get_kits_colors(players_imgs, grass_hsv, annotated_frame)
                if len(kits_colors) >= 2:  # Need at least 2 players to classify teams
                    kits_clf = get_kits_classifier(kits_colors)
                    left_team_label = get_left_team_label(players_boxes, kits_colors, kits_clf, width)
                    grass_color = get_grass_color(result.orig_img)
                    grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

                    # Get average team colors
                    teams = classify_kits(kits_clf, kits_colors)
                    for i, color in enumerate(kits_colors):
                        team = teams[i]
                        if team_colors[team] is None:
                            team_colors[team] = tuple(map(int, color))

        # Process detections
        if kits_clf is not None:  # Only process if team classification is initialized
            for box in result.boxes:
                label = int(box.cls.numpy()[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                
                # Get tracking ID
                track_id = None
                if hasattr(box, 'id'):
                    track_id = int(box.id.item())

                if label == 0:  # Player
                    if track_id is not None:
                        # Update or initialize player tracking
                        if player_tracks[track_id]["team"] is not None:
                            team = player_tracks[track_id]["team"]
                            box_color = player_tracks[track_id]["color"]
                            player_tracks[track_id]["missed_frames"] = 0
                        else:
                            # New player, determine team and color
                            player_img = result.orig_img[y1:y2, x1:x2]
                            kit_color = get_kits_colors([player_img], grass_hsv, annotated_frame)[0]
                            team = classify_kits(kits_clf, [kit_color])[0]
                            box_color = tuple(map(int, kit_color))
                            player_tracks[track_id]["team"] = team
                            player_tracks[track_id]["color"] = box_color
                            player_tracks[track_id]["missed_frames"] = 0
                    else:
                        # Fallback if no tracking ID
                        player_img = result.orig_img[y1:y2, x1:x2]
                        kit_color = get_kits_colors([player_img], grass_hsv, annotated_frame)[0]
                        team = classify_kits(kits_clf, [kit_color])[0]
                        box_color = tuple(map(int, kit_color))
                    
                    label_text = f"Player-L{track_id}" if team == left_team_label else f"Player-R{track_id}"

                elif label == 1:  # Goalkeeper
                    if x1 < 0.5 * width:
                        label = 2
                        box_color = team_colors[left_team_label]
                        label_text = f"GK-L{track_id if track_id is not None else ''}"
                    else:
                        label = 3
                        box_color = team_colors[1 if left_team_label == 0 else 0]
                        label_text = f"GK-R{track_id if track_id is not None else ''}"

                else:  # Other objects
                    label += 2
                    box_color = other_colors[str(label)]
                    base_label = ["Ball", "Main Ref", "Side Ref", "Staff"][label-4]
                    label_text = f"{base_label}{track_id if track_id is not None else ''}"
                    
                    # Store ball detections for interpolation
                    if label == 4:  # Ball
                        ball_detections.append({
                            'frame_idx': frame_count,
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2
                        })
                        last_ball_pos = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                    
                    # If ball not detected but we have previous detections, use interpolation
                    elif label != 4 and frame_count > 1 and len(ball_detections) > 1:
                        df = interpolate_ball_positions(ball_detections)
                        if frame_count in df.index:
                            ball_pos = df.loc[frame_count]
                            try:
                                x1 = int(ball_pos.x1)
                                y1 = int(ball_pos.y1)
                                x2 = int(ball_pos.x2)
                                y2 = int(ball_pos.y2)
                                # Draw interpolated ball position
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), other_colors["4"], 2)
                                cv2.putText(annotated_frame, "Ball(interp)", (x1, y1-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, other_colors["4"], 2)
                            except Exception as e:
                                print(f"Frame {frame_count}: Error processing ball position")
                                continue

                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(annotated_frame, label_text, (x1 - 30, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

            # Clean up lost tracks after certain number of frames
            for track_id in list(player_tracks.keys()):
                player_tracks[track_id]["missed_frames"] += 1
                if player_tracks[track_id]["missed_frames"] > 30:  # Remove after 30 frames
                    del player_tracks[track_id]

        output_video.write(annotated_frame)
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    print("Completed!")
    cv2.destroyAllWindows()
    output_video.release()
    cap.release()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_video>")
        sys.exit(1)

    model = YOLO(os.path.join(".", "weights", "last.pt"))
    video_path = os.path.normpath(sys.argv[1])
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    annotate_video(video_path, model)