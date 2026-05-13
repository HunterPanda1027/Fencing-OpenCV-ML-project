import cv2
import mediapipe as mp
import csv
import time
from cap_from_youtube import cap_from_youtube
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

# youtube links: 
#Left vs Kano (2025): 'https://www.youtube.com/watch?v=MVqBp6dDTXg'
#Right vs Limardo (2025): 'https://www.youtube.com/watch?v=IP1D0h0Gf4M'

youtube_url = 'https://www.youtube.com/watch?v=IP1D0h0Gf4M'
cap = cap_from_youtube(youtube_url, '1080p')

model_path = 'pose_landmarker_heavy.task'

options = PoseLandmarkerOptions(
    base_options = BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_poses=2,
    min_pose_detection_confidence=0.8,  # Initial detection sensitivity
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
    
csv_file = open('fencing_data.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp_ms',
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

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        view_mode = cv2.getTrackbarPos('View', 'Fencing Tracker')
        l_is_lefty = cv2.getTrackbarPos('Left', 'Fencing Tracker')
        r_is_lefty = cv2.getTrackbarPos('Right', 'Fencing Tracker')

        #code: fencer side part
        #fencers: r for right, l for left
        #body parts side: r for right, l for left
        #body parts: h for hips, w for wrists, s for shoulder, e for elbow, f for finger, k for knees, a for ankles


        if pose_landmarker_result.pose_landmarks:
            poses = pose_landmarker_result.pose_landmarks

            if len(poses) == 2:
                #if last_pos_l is None and last_pos_r is None:
                poses = sorted(poses, key=lambda p: p[23].x)
                fencer_l_data = poses[0]
                fencer_r_data = poses[1]
                first_frame_lock = True
                #else: 
                    #dist_0_to_l = ((poses[0][23].x - last_pos_l[0])**2 + (poses[0][23].y - last_pos_l[1])**2)**0.5
                    #dist_1_to_l = ((poses[1][23].x - last_pos_l[0])**2 + (poses[1][23].y - last_pos_l[1])**2)**0.5

                   # if dist_0_to_l < dist_1_to_l:
                      #  fencer_l_data, fencer_r_data = poses[0], poses[1]
                   # else: 
                     #   fencer_l_data, fencer_r_data = poses[1], poses[0]

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

                csv_writer.writerow([frame_timestamp_ms,
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

                h, w, _ = frame.shape
                
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

        else:
            continue

        cv2.imshow('Fencing Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
csv_file.close()
cv2.destroyAllWindows()
print("Tracking Complete. Data saved to fencing_data.csv")