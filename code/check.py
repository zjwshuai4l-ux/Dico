# import h5py

# with h5py.File('/home/zhangjw/DiCo-main/data/PARSE2022/train/h5PA000016/2022.h5', 'r') as file:
#     # 读取图像数据
#     image_data = file['/home/zhangjw/DiCo-main/data/PARSE2022/train/h5PA000016/2022.h5'][:]
#     print("Image shape:", image_data.shape)
    
#     # # 读取分割数据
#     # segmentation = file['segmentation'][:]
#     # print("Segmentation shape:", segmentation.shape)
    
#     # # 读取空间分辨率
#     # spacing = file['spacing'][()]
#     # print("Spacing:", spacing)
    
#     # # 读取原点信息
#     # origin = file['origin'][()]
#     # print("Origin:", origin)
import h5py

# 打开 HDF5 文件
with h5py.File('/home/zhangjw/DiCo-main/data/PARSE2022/train/h5PA000016/2022.h5', 'r') as f:
    # 读取 'image' 数据集
    image_data = f['image'][:]
    
    # 查看图像数据的一些基本信息
    print(f"Image data shape: {image_data.shape}")
    print(f"Image data type: {image_data.dtype}")
    print(f"Image data (first slice): {image_data[0]}")  # 打印图像数据的第一个切片（或第一个深度）
