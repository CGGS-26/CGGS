import numpy as np

def read_cameras_binary(file_path):
    with open(file_path, "rb") as f:
        try:
            # 读取相机数量
            num_cameras = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
            print(f"Number of cameras: {num_cameras}")
            cameras = {}
            
            for _ in range(num_cameras):
                # 读取相机信息
                camera_id = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
                model_id = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                width = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
                height = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
                num_params = int(np.fromfile(f, dtype=np.uint64, count=1)[0])

                # 检查参数数量是否合理
                if num_params > 20:
                    print(f"Warning: Unusually large number of parameters ({num_params}) for camera {camera_id}. Skipping.")
                    continue
                
                # 读取参数
                params = np.fromfile(f, dtype=np.float64, count=num_params)
                cameras[camera_id] = (model_id, width, height, params)
                
                print(f"Camera ID: {camera_id}, Model ID: {model_id}, "
                      f"Width: {width}, Height: {height}, Params: {params}")
            
            return cameras
        
        except Exception as e:
            print(f"Error reading binary file: {e}")
            return None

# 调用函数读取相机文件
cameras = read_cameras_binary("outputs/local/colmap/sparse/0/cameras.bin")
