import time
from tracker import generate_data

video_queue = [
    {
        "name": "Kano (2025)",
        "url" : 'https://www.youtube.com/watch?v=MVqBp6dDTXg',
        "side": 0,
        "is_lefty": 0,
    },
    {
        "name": "Limardo (2025)",
        "url" : 'https://www.youtube.com/watch?v=IP1D0h0Gf4M',
        "side": 1,
        "is_lefty": 0,

    },
    {
        "name": "Koshman (2025)",
        "url" : 'https://www.youtube.com/watch?v=BUp2XJPfGMM',
        "side": 1,
        "is_lefty": 0,
    },
    {
        "name": "Bida (2019)",
        "url" : 'https://www.youtube.com/watch?v=KQOebPZd8nU',
        "side": 1,
        "is_lefty": 0,
    },
    {
        "name": "Loyola (2025)",
        "url" : 'https://www.youtube.com/watch?v=rKGKpmG89Yc',
        "side": 1,
        "is_lefty": 0,
    },
    {
        "name": "Borel (2022)",
        "url" : 'https://www.youtube.com/watch?v=l0gb5h83N9Y',
        "side": 0,
        "is_lefty": 0,
    },
]

print(f"Generating data for {len(video_queue)} videos...")
start_time = time.time()

for index, video in enumerate (video_queue, 1):
    print(f"\n==================================================")
    print(f"PROCESSING BOUT {index} OF {len(video_queue)}")
    print(f"==================================================")

    try:
        generate_data(video["url"], video["name"], video["side"], video["is_lefty"])
    except Exception as e:
        print(f"ERROR : Error processing {video['name']}: {e}")
        print("Moving to next video in the queue...")
        continue

total_duration = (time.time() - start_time)/ 60
print(f"\n All videos processed successfully in {total_duration: .1f} minutes!")