import cv2
import mediapipe as mp
import csv
import time
from cap_from_youtube import cap_from_youtube
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

youtube_url = 'https://www.youtube.com/watch?v=MVqBp6dDTXg'
cap = cap_from_youtube(youtube_url, '1080p')

model_path = 'pose_landmarker_heavy.task'

options = PoseLandmarkerOptions(
    base_options = BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_poses=2,
    min_pose_detection_confidence=0.7,  # Initial detection sensitivity
    min_pose_presence_confidence=0.5,   # Sensitivity to keep tracking the pose
    min_tracking_confidence=0.5  #temporal consistency sensitivity
)

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
                     'lre_x', 'lre_y', 'lre_z',
                     'lrh_x', 'lrh_y', 'lrh_z',
                     'lrf_x', 'lrf_y', 'lrf_z',
                     
                     'rla_x', 'rla_y', 'rla_z',
                     'rra_x', 'rra_y', 'rra_z',
                     'rlk_x', 'rlk_y', 'rlk_z',
                     'rrk_x', 'rrk_y', 'rrk_z',
                     'rlh_x', 'rlh_y', 'rlh_z',
                     'rrh_x', 'rrh_y', 'rrh_z',
                     'rls_x', 'rls_y', 'rls_z',
                     'rrs_x', 'rrs_y', 'rrs_z',
                     'rre_x', 'rre_y', 'rre_z',
                     'rrh_x', 'rrh_y', 'rrh_z',
                     'rrf_x', 'rrf_y', 'rrf_z'])

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

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
                lrw = fencer_l_data[16]
                lrs = fencer_l_data[12]
                lls = fencer_l_data[11]
                lre = fencer_l_data[14]
                lrf = fencer_l_data[20]
                lrk = fencer_l_data[26]
                llk = fencer_l_data[25]
                lra = fencer_l_data[28]
                lla = fencer_l_data[27]

                rlh = fencer_r_data[23]
                rrh = fencer_r_data[24]
                rrw = fencer_r_data[16]
                rrs = fencer_r_data[12]
                rls = fencer_r_data[11]
                rre = fencer_r_data[14]
                rrf = fencer_r_data[20]
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
                                    lre.x, lre.y, lre.z,
                                    lrh.x, lrh.y, lrh.z,
                                    lrf.x, lrf.y, lrf.z,
                                    
                                    rla.x, rla.y, rla.z,
                                    rra.x, rra.y, rra.z,
                                    rlk.x, rlk.y, rlk.z,
                                    rrk.x, rrk.y, rrk.z,
                                    rlh.x, rlh.y, rlh.z,
                                    rrh.x, rrh.y, rrh.z,
                                    rls.x, rls.y, rls.z,
                                    rrs.x, rrs.y, rrs.z,
                                    rre.x, rre.y, rre.z,
                                    rrh.x, rrh.y, rrh.z,
                                    rrf.x, rrf.y, rrf.z])

                h, w, _ = frame.shape
                
                #left fencer
                llhx, llhy = int(llh.x * w), int(llh.y * h)
                cv2.circle(frame, (llhx, llhy), 8, (0, 0, 255), -1)

                lrhx, lrhy = int(lrh.x * w), int(lrh.y * h)
                cv2.circle(frame, (lrhx, lrhy), 8, (0, 0, 255), -1)

                lrwx, lrwy = int(lrw.x * w), int(lrw.y * h)
                cv2.circle(frame, (lrwx, lrwy), 8, (0, 0, 255), -1)

                lrsx, lrsy = int(lrs.x * w), int(lrs.y * h)
                cv2.circle(frame, (lrsx, lrsy), 8, (0, 0, 255), -1)

                llsx, llsy = int(lls.x * w), int(lls.y * h)
                cv2.circle(frame, (llsx, llsy), 8, (0, 0, 255), -1)

                lrex, lrey = int(lre.x * w), int(lre.y * h)
                cv2.circle(frame, (lrex, lrey), 8, (0, 0, 255), -1)

                lrfx, lrfy = int(lrf.x * w), int(lrf.y * h)
                cv2.circle(frame, (lrfx, lrfy), 8, (0, 0, 255), -1)

                lrkx, lrky = int(lrk.x * w), int(lrk.y * h)
                cv2.circle(frame, (lrkx, lrky), 8, (0, 0, 255), -1)

                llkx, llky = int(llk.x * w), int(llk.y * h)
                cv2.circle(frame, (llkx, llky), 8, (0, 0, 255), -1)

                lrax, lray = int(lra.x * w), int(lra.y * h)
                cv2.circle(frame, (lrax, lray), 8, (0, 0, 255), -1)

                llax, llay = int(lla.x * w), int(lla.y * h)
                cv2.circle(frame, (llax, llay), 8, (0, 0, 255), -1)

                #right fencer
                rlhx, rlhy = int(rlh.x * w), int(rlh.y * h)
                cv2.circle(frame, (rlhx, rlhy), 8, (0, 255, 0), -1)

                rrhx, rrhy = int(rrh.x * w), int(rrh.y * h)
                cv2.circle(frame, (rrhx, rrhy), 8, (0, 255, 0), -1)

                rrwx, rrwy = int(rrw.x * w), int(rrw.y * h)
                cv2.circle(frame, (rrwx, rrwy), 8, (0, 255, 0), -1)

                rrsx, rrsy = int(rrs.x * w), int(rrs.y * h)
                cv2.circle(frame, (rrsx, rrsy), 8, (0, 255, 0), -1)

                rlsx, rlsy = int(rls.x * w), int(rls.y * h)
                cv2.circle(frame, (rlsx, rlsy), 8, (0, 255, 0), -1)

                rrex, rrey = int(rre.x * w), int(rre.y * h)
                cv2.circle(frame, (rrex, rrey), 8, (0, 255, 0), -1)

                rrfx, rrfy = int(rrf.x * w), int(rrf.y * h)
                cv2.circle(frame, (rrfx, rrfy), 8, (0, 255, 0), -1)

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