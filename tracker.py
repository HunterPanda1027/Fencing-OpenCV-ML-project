import cv2
import mediapipe as mp
import csv
import time
from cap_from_youtube import cap_from_youtube
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import numpy as np

def generate_data(youtube_url, match_name):
    """Processes a single video and saves data to a unique CSV"""

    cap = cap_from_youtube(youtube_url, 'best')

    model_path = 'pose_landmarker_heavy.task'

    options = PoseLandmarkerOptions(
        base_options = BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=2,
        min_pose_detection_confidence=0.7,  # Initial detection sensitivity
        min_pose_presence_confidence=0.8,   # Sensitivity to keep tracking the pose
        min_tracking_confidence=0.5  #temporal consistency sensitivity
    )

    def nothing(x):
        pass

    cv2.namedWindow('Fencing Tracker')

    cv2.createTrackbar('View', 'Fencing Tracker', 3, 3, nothing)
    cv2.createTrackbar('Left', 'Fencing Tracker', 0, 1, nothing)
    cv2.createTrackbar('Right', 'Fencing Tracker', 0, 1, nothing)

    #Making sure that the first fencer on the left will be the left fencer at all time
    first_frame_lock = False
    left_fencer_id = None

    last_pos_l = None
    last_pos_r = None
        
    csv_file = open(f'fencing_data_{match_name}.csv', mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp_ms', 'cam_shift'
                        'lla_x', 'lla_y', 'lla_z',
                        'lra_x', 'lra_y', 'lra_z',
                        'llk_x', 'llk_y', 'llk_z',
                        'lrk_x', 'lrk_y', 'lrk_z',
                        'llh_x', 'llh_y', 'llh_z',
                        'lrh_x', 'lrh_y', 'lrh_z',
                        'lls_x', 'lls_y', 'lls_z',
                        'lrs_x', 'lrs_y', 'lrs_z',
                        'le_x', 'le_y', 'le_z',
                        'lw_x', 'lw_y', 'lw_z',
                        'lf_x', 'lf_y', 'lf_z',
                        
                        'rla_x', 'rla_y', 'rla_z',
                        'rra_x', 'rra_y', 'rra_z',
                        'rlk_x', 'rlk_y', 'rlk_z',
                        'rrk_x', 'rrk_y', 'rrk_z',
                        'rlh_x', 'rlh_y', 'rlh_z',
                        'rrh_x', 'rrh_y', 'rrh_z',
                        'rls_x', 'rls_y', 'rls_z',
                        'rrs_x', 'rrs_y', 'rrs_z',
                        're_x', 're_y', 're_z',
                        'rw_x', 'rw_y', 'rw_z',
                        'rf_x', 'rf_y', 'rf_z'])

    prev_gray = None
    prev_fencer_x = None
    prev_timestamp = None

    frame_counter = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_counter += 1
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            h, w, _ = frame.shape
        
            mediapipe_frame = frame.copy()
            bottom_mask = int(h*0.85)
            mediapipe_frame[bottom_mask: h, : ] = 0
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mediapipe_frame)

            pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            view_mode = cv2.getTrackbarPos('View', 'Fencing Tracker')
            l_is_lefty = cv2.getTrackbarPos('Left', 'Fencing Tracker')
            r_is_lefty = cv2.getTrackbarPos('Right', 'Fencing Tracker')

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_timestamp = frame_timestamp_ms
            dx_cam = 0.0

            mask = np.ones(gray.shape, dtype=np.uint8) * 255
            mask[int(h*0.75):h, :] = 0
            mask[0:int(h * 0.15), 0:int(w * 0.15)] = 0

            if pose_landmarker_result.pose_landmarks:
                poses = pose_landmarker_result.pose_landmarks
                if len(poses) == 2:
                    poses = sorted(poses, key=lambda p: p[23].x)
                    fencer_l_data = poses[0]
                    fencer_r_data = poses[1]
                    first_frame_lock = True

                    #Mask for the left fencer
                    fencer_mid_x_left = int(fencer_l_data[23].x * w)
                    xmin_block_left = max(0, fencer_mid_x_left - int(0.08 * w))
                    xmax_block_left = min(w, fencer_mid_x_left + int(0.15 * w))
                    ymin_block_left = max(0, int(fencer_l_data[0].y * h) - int(0.07 * h))
                    ymax_block_left = min(h, int(fencer_l_data[27].y * h) + int(0.07 * h))
                    mask[ymin_block_left:ymax_block_left, xmin_block_left:xmax_block_left] = 0

                    #for the right fencer
                    fencer_mid_x_right = int(fencer_r_data[23].x * w)
                    xmin_block_right = max(0, fencer_mid_x_right - int(0.15 * w))
                    xmax_block_right = min(w, fencer_mid_x_right + int(0.08 * w))
                    ymin_block_right = max(0, int(fencer_r_data[0].y * h) - int(0.07 * h))
                    ymax_block_right = min(h, int(fencer_r_data[27].y * h) + int(0.07 * h))
                    mask[ymin_block_right:ymax_block_right, xmin_block_right:xmax_block_right] = 0

            

                    if prev_gray is not None:
                        # Step A: Find sharp features anywhere in the current frame (strip edges, machine lights)
                        # Because the background is black, it will naturally find the strip and machine!
                        feat_prev = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10, mask=mask)
                        
                        if feat_prev is not None:
                            # Step B: Track where those points moved in the new frame
                            feat_next, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, feat_prev, None)
                            
                            valid_prev = feat_prev[status == 1]
                            valid_next = feat_next[status == 1]
                            
                            if len(valid_next) >= 4: # We need at least 4 points to calculate background movement
                                # Step C: The RANSAC Magic. It filters out the fencers and finds the true camera shift.
                                matrix, inliers = cv2.estimateAffinePartial2D(valid_prev, valid_next, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                                
                                if matrix is not None:
                                    # dx_cam is the horizontal translation component of the camera movement matrix
                                    dx_cam = matrix[0, 2]

                                    # --- DEBUG DOTS ---
                                    # Loop through tracked features to verify what the math sees
                                    for i, (p_prev, p_next) in enumerate(zip(valid_prev, valid_next)):
                                        pt_x, pt_y = map(int, p_next)
                                        if inliers[i] == 1:
                                            # Safe Background Anchor point (Strip/Machine) -> GREEN
                                            cv2.circle(frame, (pt_x, pt_y), 3, (0, 255, 0), -1)
                                        else:
                                            # Moving Object Point (Fencer) -> RED (Ignored by math)
                                            cv2.circle(frame, (pt_x, pt_y), 4, (0, 0, 255), -1)
                            

                    #code: fencer side part
                    #fencers: r for right, l for left
                    #body parts side: r for right, l for left
                    #body parts: h for hips, w for wrists, s for shoulder, e for elbow, f for finger, k for knees, a for ankles

                    poses = sorted(poses, key=lambda p: p[23].x)
                    fencer_l_data = poses[0]
                    fencer_r_data = poses[1]
                    first_frame_lock = True

                    last_pos_l = (fencer_l_data[23].x, fencer_l_data[23].y)
                    last_pos_r = (fencer_r_data[23].x, fencer_r_data[23].y)

                    llh = fencer_l_data[23]
                    lrh = fencer_l_data[24]
                    lw = fencer_l_data[15] if l_is_lefty == 1 else fencer_l_data[16]
                    lrs = fencer_l_data[12]
                    lls = fencer_l_data[11]
                    le = fencer_l_data[13] if l_is_lefty == 1 else fencer_l_data[14]
                    lf = fencer_l_data[19] if l_is_lefty == 1 else fencer_l_data[20]
                    lrk = fencer_l_data[26]
                    llk = fencer_l_data[25]
                    lra = fencer_l_data[28]
                    lla = fencer_l_data[27]

                    rlh = fencer_r_data[23]
                    rrh = fencer_r_data[24]
                    rw = fencer_r_data[15] if r_is_lefty == 1 else fencer_r_data[16]
                    rrs = fencer_r_data[12]
                    rls = fencer_r_data[11]
                    re = fencer_r_data[13] if r_is_lefty == 1 else fencer_r_data[14]
                    rf = fencer_r_data[19] if r_is_lefty == 1 else fencer_r_data[20]
                    rrk = fencer_r_data[26]
                    rlk = fencer_r_data[25]
                    rra = fencer_r_data[28]
                    rla = fencer_r_data[27]

                    csv_writer.writerow([frame_timestamp_ms, dx_cam,
                                        lla.x, lla.y, lla.z,
                                        lra.x, lra.y, lra.z,
                                        llk.x, llk.y, llk.z,
                                        lrk.x, lrk.y, lrk.z,
                                        llh.x, llh.y, llh.z,
                                        lrh.x, lrh.y, lrh.z,
                                        lls.x, lls.y, lls.z,
                                        lrs.x, lrs.y, lrs.z,
                                        le.x, le.y, le.z,
                                        lw.x, lw.y, lw.z,
                                        lf.x, lf.y, lf.z,
                                        
                                        rla.x, rla.y, rla.z,
                                        rra.x, rra.y, rra.z,
                                        rlk.x, rlk.y, rlk.z,
                                        rrk.x, rrk.y, rrk.z,
                                        rlh.x, rlh.y, rlh.z,
                                        rrh.x, rrh.y, rrh.z,
                                        rls.x, rls.y, rls.z,
                                        rrs.x, rrs.y, rrs.z,
                                        re.x, re.y, re.z,
                                        rw.x, rw.y, rw.z,
                                        rf.x, rf.y, rf.z])
                    
                    #left fencer
                    if view_mode in [1,3]: 

                        llhx, llhy = int(llh.x * w), int(llh.y * h)
                        cv2.circle(frame, (llhx, llhy), 8, (0, 0, 255), -1)

                        lrhx, lrhy = int(lrh.x * w), int(lrh.y * h)
                        cv2.circle(frame, (lrhx, lrhy), 8, (0, 0, 255), -1)

                        lwx, lwy = int(lw.x * w), int(lw.y * h)
                        cv2.circle(frame, (lwx, lwy), 8, (0, 0, 255), -1)

                        lrsx, lrsy = int(lrs.x * w), int(lrs.y * h)
                        cv2.circle(frame, (lrsx, lrsy), 8, (0, 0, 255), -1)

                        llsx, llsy = int(lls.x * w), int(lls.y * h)
                        cv2.circle(frame, (llsx, llsy), 8, (0, 0, 255), -1)

                        lex, ley = int(le.x * w), int(le.y * h)
                        cv2.circle(frame, (lex, ley), 8, (0, 0, 255), -1)

                        lfx, lfy = int(lf.x * w), int(lf.y * h)
                        cv2.circle(frame, (lfx, lfy), 8, (0, 0, 255), -1)

                        lrkx, lrky = int(lrk.x * w), int(lrk.y * h)
                        cv2.circle(frame, (lrkx, lrky), 8, (0, 0, 255), -1)

                        llkx, llky = int(llk.x * w), int(llk.y * h)
                        cv2.circle(frame, (llkx, llky), 8, (0, 0, 255), -1)

                        lrax, lray = int(lra.x * w), int(lra.y * h)
                        cv2.circle(frame, (lrax, lray), 8, (0, 0, 255), -1)

                        llax, llay = int(lla.x * w), int(lla.y * h)
                        cv2.circle(frame, (llax, llay), 8, (0, 0, 255), -1)

                    if view_mode in [2,3]:

                        #right fencer
                        rlhx, rlhy = int(rlh.x * w), int(rlh.y * h)
                        cv2.circle(frame, (rlhx, rlhy), 8, (0, 255, 0), -1)

                        rrhx, rrhy = int(rrh.x * w), int(rrh.y * h)
                        cv2.circle(frame, (rrhx, rrhy), 8, (0, 255, 0), -1)

                        rwx, rwy = int(rw.x * w), int(rw.y * h)
                        cv2.circle(frame, (rwx, rwy), 8, (0, 255, 0), -1)

                        rrsx, rrsy = int(rrs.x * w), int(rrs.y * h)
                        cv2.circle(frame, (rrsx, rrsy), 8, (0, 255, 0), -1)

                        rlsx, rlsy = int(rls.x * w), int(rls.y * h)
                        cv2.circle(frame, (rlsx, rlsy), 8, (0, 255, 0), -1)

                        rex, rey = int(re.x * w), int(re.y * h)
                        cv2.circle(frame, (rex, rey), 8, (0, 255, 0), -1)

                        rfx, rfy = int(rf.x * w), int(rf.y * h)
                        cv2.circle(frame, (rfx, rfy), 8, (0, 255, 0), -1)

                        rrkx, rrky = int(rrk.x * w), int(rrk.y * h)
                        cv2.circle(frame, (rrkx, rrky), 8, (0, 255, 0), -1)

                        rlkx, rlky = int(rlk.x * w), int(rlk.y * h)
                        cv2.circle(frame, (rlkx, rlky), 8, (0, 255, 0), -1)

                        rrax, rray = int(rra.x * w), int(rra.y * h)
                        cv2.circle(frame, (rrax, rray), 8, (0, 255, 0), -1)

                        rlax, rlay = int(rla.x * w), int(rla.y * h)
                        cv2.circle(frame, (rlax, rlay), 8, (0, 255, 0), -1)

                    
            prev_gray = gray.copy()
            debug_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # Display raw camera frame shift value on video overlay
            cv2.putText(frame, f"Cam Shift: {dx_cam:.1f} px", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow('Fencing Tracker', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('f'):
                print("Fast Forward 5 seconds.")
                frame_counter += 150
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

                #wiping tracking history to prevent velocity spikes etc.
                prev_gray = None
                prev_fencer_x = None
                prev_timestamp = None


    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print("Tracking Complete. Data saved to fencing_data.csv")




#Allows for more testing
if __name__ == "__main__":
    test_url = 'https://www.youtube.com/watch?v=MVqBp6dDTXg'
    test_name = "Left vs Kano (2025)"
    generate_data(test_url, test_name)