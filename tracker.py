import cv2
import mediapipe as mp
import csv
import time
from cap_from_youtube import cap_from_youtube
#from mediapipe.tasks import python
#from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

youtube_url = 'https://www.youtube.com/watch?v=MVqBp6dDTXg'
cap = cap_from_youtube(youtube_url, '1080p')

model_path = 'pose_landmarker_heavy.task'

options = PoseLandmarkerOptions(
    base_options = BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_poses=2
)

#Making sure that the first fencer on the left will be the left fencer at all time
first_frame_lock = False
left_fencer_id = None

last_pos_l = None
last_pos_r = None
    
csv_file = open('fencing_data.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp_ms', 'hip_x', 'hip_y', 'hip_z'])

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
                if last_pos_l is None:
                    poses = sorted(poses, key=lambda p: p[23].x)
                    fencer_l_data = poses[0]
                    fencer_r_data = poses[1]
                    first_frame_lock = True
                else: 
                    dist_0_to_l = ((poses[0][23].x - last_pos_l[0])**2 + (poses[0][23].y - last_pos_l[1])**2)**0.5
                    dist_1_to_l = ((poses[1][23].x - last_pos_l[0])**2 + (poses[1][23].y - last_pos_l[1])**2)**0.5

                    if dist_0_to_l < dist_1_to_l:
                        fencer_l_data, fencer_r_data = poses[0], poses[1]
                    else: 
                        fencer_l_data, fencer_r_data = poses[1], poses[0]

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

                csv_writer.writerow([frame_timestamp_ms, llh.x, llh.y, llh.z, lrw.x, lrw.y, lrw.z])

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