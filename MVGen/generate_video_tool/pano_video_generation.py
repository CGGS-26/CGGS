import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import lib.Equirec2Perspec as E2P
import lib.multi_Perspec2Equirec as m_P2E
import uuid
from tqdm import tqdm
import imageio

def generate_video(image_paths, out_dir, FOV, deg, gen_video=False, save_frames=False):
    # Load images from the provided paths
    pers = [cv2.imread(image_path) for image_path in image_paths]
    if any(img is None for img in pers):
        print("Error: One or more images could not be loaded.")
        return

    # Generate panoramic image
    pers_matrix = [[FOV, deg * i, 0] for i in range(int(360 / deg))]
    ee = m_P2E.Perspective(pers, pers_matrix)
    new_pano = ee.GetEquirec(2048, 4096)
    cv2.imwrite(os.path.join(out_dir, 'pano.png'), new_pano.astype(np.uint8)[540:-540])

    if not gen_video:
        return

    # Set video parameters
    equ = E2P.Equirectangular(new_pano)
    fov = 60
    # video_size = (500, 500)
    video_size = (512, 512)
    img = equ.GetPerspective(fov, 0, 0, 512, 512)
    
    margin = 0
    if margin > 0:
        img = img[margin:-margin]
    size = (img.shape[1], img.shape[0])

    tmp_video_path = '/tmp/' + str(uuid.uuid4()) + '.mp4'
    save_video_path = os.path.join(out_dir, 'video.mp4')
    out = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 60, size)

    interval_deg = 1
    num_frames = int(360 / interval_deg)
    image_save_path = os.path.join(out_dir, 'images')
    os.makedirs(image_save_path, exist_ok=True)

    for i in range(num_frames):
        frame_deg = i * interval_deg
        img = equ.GetPerspective(fov, frame_deg, 0, size[1], size[0])
        if margin > 0:
            img = img[margin:-margin]
        img = np.clip(img, 0, 255).astype(np.uint8)
        # Optionally save each frame as an image
        if save_frames:
            frame_filename = os.path.join(image_save_path, f'{i:04d}.png')
            cv2.imwrite(frame_filename, img)

        out.write(img)

    out.release()
    # cv2.destroyAllWindows()
    print(f"Video saved at {save_video_path}")


# import cv2
# import numpy as np
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# import lib.Equirec2Perspec as E2P
# import lib.multi_Perspec2Equirec as m_P2E
# from tqdm import tqdm
# import imageio

# def generate_video(image_paths, out_dir, FOV, deg, gen_video=False, save_frames=False):
#     # Load images from the provided paths
#     pers = [cv2.imread(image_path) for image_path in image_paths]
#     if any(img is None for img in pers):
#         print("Error: One or more images could not be loaded.")
#         return

#     # Generate panoramic image
#     pers_matrix = [[FOV, deg * i, 0] for i in range(int(360 / deg))]
#     ee = m_P2E.Perspective(pers, pers_matrix)
#     new_pano = ee.GetEquirec(2048, 4096)
#     cv2.imwrite(os.path.join(out_dir, 'pano.png'), new_pano.astype(np.uint8)[540:-540])

#     if not gen_video:
#         return

#     # Set video parameters
#     equ = E2P.Equirectangular(new_pano)
#     fov = 90
#     video_size = (450, 600)

#     # Prepare to save video frames
#     interval_deg = 1
#     num_frames = int(360 / interval_deg)  # 1 degree interval
#     images = []

#     for i in tqdm(range(num_frames), desc="Generating video frames"):
#         frame_deg = i * 1  # interval_deg is 1
#         img = equ.GetPerspective(fov, frame_deg, 0, video_size[1], video_size[0])
#         img = np.clip(img, 0, 255).astype(np.uint8)
#         images.append(img)

#         # Optionally save each frame as an image
#         if save_frames:
#             frame_filename = os.path.join(out_dir, 'images', f'{i:04d}.png')
#             os.makedirs(os.path.dirname(frame_filename), exist_ok=True)
#             cv2.imwrite(frame_filename, img)

#     # Use imageio to write the video
#     save_video_path = os.path.join(out_dir, 'video.mp4')
#     imageio.mimwrite(save_video_path, images, fps=60, quality=10)

#     print(f"Video saved at {save_video_path}.")
