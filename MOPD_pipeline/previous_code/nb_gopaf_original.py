# Auto-generated from /data/private/user2/workspace/ocean/oneocean(iros-2026-code)/MOPD_pipeline/previous_code/gopaf_original.ipynb
# Generated at 2026-02-23 02:12:50
# Note: notebook outputs are omitted; IPython magics are commented out.

# %% [cell 0]
import xarray as xr

# 加载合并后的数据集
combined_data = xr.open_dataset("cmems_detailed_uv_output_data.nc")

# 打印数据集的概况信息
print(combined_data)

# 打印数据集中的前几行数据（可以打印变量、时间、深度等信息）
# 使用 .isel() 方法选择前几行数据进行打印
print(combined_data.isel(time=0))  # 打印时间维度为第一个时间点的所有数据

print("-------------")
# 打印某个变量的数据值的前几行
print(combined_data['uo'].isel(time=0))  # 查看 'uo' (东向海水速度) 在第一个时间点的数据

# %% [cell 1]
import xarray as xr

# 加载两个数据集
basic_phy_data = xr.open_dataset("cmems_basic_phy_output_data.nc")  # 基础物理变量数据集
detailed_uv_data = xr.open_dataset("cmems_detailed_uv_output_data.nc")  # 详细流速数据集

# 获取两个数据集中的 'uo' 和 'vo' 变量
uo_basic = basic_phy_data['uo']
vo_basic = basic_phy_data['vo']
uo_detailed = detailed_uv_data['uo']
vo_detailed = detailed_uv_data['vo']

# 打印 'uo' 和 'vo' 的完整属性
print("基础物理变量数据集中的 'uo' 属性:")
print(uo_basic.attrs)

print("\n详细流速数据集中的 'uo' 属性:")
print(uo_detailed.attrs)

print("\n基础物理变量数据集中的 'vo' 属性:")
print(vo_basic.attrs)

print("\n详细流速数据集中的 'vo' 属性:")
print(vo_detailed.attrs)

# 打印 'uo' 和 'vo' 的更多数值内容
print("\n基础物理变量数据集中的 'uo' 更多数值:")
print(uo_basic.isel(latitude=slice(0, 10), longitude=slice(0, 10)))

print("\n详细流速数据集中的 'uo' 更多数值:")
print(uo_detailed.isel(latitude=slice(0, 10), longitude=slice(0, 10)))

print("\n基础物理变量数据集中的 'vo' 更多数值:")
print(vo_basic.isel(latitude=slice(0, 10), longitude=slice(0, 10)))

print("\n详细流速数据集中的 'vo' 更多数值:")
print(vo_detailed.isel(latitude=slice(0, 10), longitude=slice(0, 10)))

# 计算 'uo' 和 'vo' 的统计量（最大值、最小值、均值）
print("\n基础物理变量数据集中的 'uo' 统计量:")
print(uo_basic.min(), uo_basic.max(), uo_basic.mean())

print("\n详细流速数据集中的 'uo' 统计量:")
print(uo_detailed.min(), uo_detailed.max(), uo_detailed.mean())

print("\n基础物理变量数据集中的 'vo' 统计量:")
print(vo_basic.min(), vo_basic.max(), vo_basic.mean())

print("\n详细流速数据集中的 'vo' 统计量:")
print(vo_detailed.min(), vo_detailed.max(), vo_detailed.mean())

# %% [cell 2]
import xarray as xr
import matplotlib.pyplot as plt

def read_and_visualize_combined_data(file_path):
    # Read the merged NetCDF file
    ds = xr.open_dataset(file_path)

    # Print example data
    print("Combined dataset contents:")
    print(ds)

    # Print first few rows of 'uo' and 'vo'
    print("\nFirst few rows of 'uo' variable:")
    print(ds['uo'].isel(latitude=slice(0, 5), longitude=slice(0, 5)))

    print("\nFirst few rows of 'vo' variable:")
    print(ds['vo'].isel(latitude=slice(0, 5), longitude=slice(0, 5)))

    # Visualize 'uo' and 'vo' data at the first time step
    time_step = 0
    uo_data = ds['uo'].isel(time=time_step, depth=0)
    vo_data = ds['vo'].isel(time=time_step, depth=0)

    # Plot 'uo'
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(uo_data.longitude, uo_data.latitude, uo_data, shading='auto')
    plt.colorbar(label='Eastward Sea Water Velocity (m/s)')
    plt.title('Eastward Sea Water Velocity (uo)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    # Plot 'vo'
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(vo_data.longitude, vo_data.latitude, vo_data, shading='auto')
    plt.colorbar(label='Northward Sea Water Velocity (m/s)')
    plt.title('Northward Sea Water Velocity (vo)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

# Example call
if __name__ == "__main__":
    file_path = "./Data/GCPAF/combined_comprehensive_output_data.nc"
    read_and_visualize_combined_data(file_path)

# %% [cell 3]
import xarray as xr

def read_and_print_values(file_path):
    # Read the NetCDF file
    ds = xr.open_dataset(file_path)

    # Get the first few rows of 'uo' and 'vo' (Eastward and Northward velocity)
    uo_values = ds['uo'].isel(latitude=slice(0, 5), longitude=slice(0, 5)).values
    vo_values = ds['vo'].isel(latitude=slice(0, 5), longitude=slice(0, 5)).values

    # Print the values
    print("\nValues of 'uo' (Eastward Sea Water Velocity) at first 5 latitudes and longitudes:")
    print(uo_values)

    print("\nValues of 'vo' (Northward Sea Water Velocity) at first 5 latitudes and longitudes:")
    print(vo_values)

# Example call
if __name__ == "__main__":
    file_path = "./Data/GCPAF/combined_comprehensive_output_data.nc"
    read_and_print_values(file_path)

# %% [cell 4]
import xarray as xr

def read_and_print_variable_shapes(file_path):
    # Load the NetCDF file
    ds = xr.open_dataset(file_path)
    
    # Iterate over each variable in the dataset and print its shape
    for var_name in ds.data_vars:
        data = ds[var_name]
        print(f"Variable '{var_name}' shape: {data.shape}")

# Example call
if __name__ == "__main__":
    file_path = "./Data/GCPAF/combined_comprehensive_output_data.nc"
    read_and_print_variable_shapes(file_path)

# %% [cell 5]
import rasterio
import os

def get_saved_geotiff_shapes(save_folder_path):
    file_path = os.path.join(save_folder_path, 'filtered_data.tif')
    
    if not os.path.exists(file_path):
        print("No saved GeoTIFF file found in the specified folder.")
        return
    
    with rasterio.open(file_path) as dataset:
        data = dataset.read()
        data_shape = data.shape
        print(f"Data dimensions: {data_shape}")
        print(f"File: {file_path}, Shape: {data_shape}")

save_path = "/Users/a1234/Desktop/workspace/*RL_AUV_2024/output"

get_saved_geotiff_shapes(save_path)

# %% [cell 6]

import copernicusmarine

# 登录 CMEMS 账号
copernicusmarine.login(
    username='sliu42',  # 替换为您的 CMEMS 用户名
    password='Lsj200106013519'   # 替换为您的 CMEMS 密码
)
# 修改深度范围以获取不同深度的数据
copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
    variables=["so", "thetao", "uo", "vo", "zos"],
    minimum_longitude=-180,
    maximum_longitude=179.91668701171875,
    minimum_latitude=-80,
    maximum_latitude=90,
    start_datetime="2024-11-25T23:00:00",
    end_datetime="2024-11-25T23:00:00",
    minimum_depth=0.0,  # 设置最小深度
    maximum_depth=100.0,  # 设置最大深度，获取从表层到 100 米深的海洋环境
    output_filename="cmems_depth_range_output_data.nc"
)

# %% [cell 7]
import xarray as xr
import matplotlib.pyplot as plt
import os
from PIL import Image

def visualize_each_variable(dataset, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for var_name in dataset.data_vars:
        data = dataset[var_name]
        plt.figure(figsize=(16, 10), dpi=200)  # Larger figure size and higher DPI for better clarity
        plt.pcolormesh(data.longitude, data.latitude, data.isel(time=0, depth=0), shading='auto')
        plt.colorbar(label=f'{var_name} ({data.attrs.get("units", "unknown unit")})')
        plt.title(f'{data.attrs.get("long_name", var_name)}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        file_path = os.path.join(output_directory, f"{var_name}.png")
        plt.savefig(file_path)
        plt.close()

def create_combined_image(image_directory, output_path):
    image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith('.png')]
    images = [Image.open(file) for file in image_files]

    widths, heights = zip(*(img.size for img in images))
    total_width = max(widths) * 2
    total_height = (len(images) + 1) // 2 * max(heights)

    combined_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    x_offset = 0
    y_offset = 0
    for i, img in enumerate(images):
        combined_image.paste(img, (x_offset, y_offset))
        if i % 2 == 1:
            x_offset = 0
            y_offset += img.height
        else:
            x_offset += img.width

    combined_image.save(output_path)

if __name__ == "__main__":
    input_file_path = "./Data/GCPAF/combined_comprehensive_output_data.nc"
    output_images_directory = "./output"
    combined_image_output_path = "./output/combined_visualization.png"

    ds = xr.open_dataset(input_file_path)
    visualize_each_variable(ds, output_images_directory)
    create_combined_image(output_images_directory, combined_image_output_path)

    print(f"Combined visualization saved at: {combined_image_output_path}")

