# Auto-generated from /data/private/user2/workspace/ocean/oneocean(iros-2026-code)/MOPD_pipeline/previous_code/buildmap.ipynb
# Generated at 2026-02-23 02:12:50
# Note: notebook outputs are omitted; IPython magics are commented out.

# %% [cell 0]
import rasterio

with rasterio.open('./output/filtered_data.tif') as src:
    print(src.crs)  # 查看坐标系
    print(src.bounds)  # 查看地理边界

# %% [cell 1]
import xarray as xr

ds = xr.open_dataset('./Data/GCPAF/combined_gcpaf_data.nc')
print(ds)  # 查看坐标信息

# %% [cell 2]
import rasterio
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

# 读取地形文件
with rasterio.open('./output/filtered_data.tif') as src:
    elevation = src.read(1)  # 读取地形数据
    bounds = src.bounds  # 获取地形边界
    transform = src.transform  # 获取仿射变换，用于定位经纬度
    resolution = src.res  # 获取分辨率
    # 计算地形数据的经纬度网格
    lons = np.linspace(bounds.left, bounds.right, src.width)
    lats = np.linspace(bounds.top, bounds.bottom, src.height)

# 读取水体数据
ds = xr.open_dataset('./Data/GCPAF/combined_gcpaf_data.nc')

# 提取原始水体数据的纬度、经度
original_lats = ds['latitude'].values
original_lons = ds['longitude'].values

# 创建网格，用于插值
lon_grid, lat_grid = np.meshgrid(original_lons, original_lats)

# 将地形的纬度和经度展开，生成插值目标网格
target_lon_grid, target_lat_grid = np.meshgrid(lons, lats)

# 对所有水体变量进行插值
variables_to_interpolate = ['so', 'thetao', 'uo', 'vo', 'zos', 'utide', 'utotal', 'vtide', 'vtotal']
interpolated_variables = {}

for var in variables_to_interpolate:
    print(f"Interpolating variable: {var}")
    data = ds[var].isel(time=0, depth=0).values  # 选择特定时间和深度的数据
    rbf_interpolator = Rbf(lon_grid.flatten(), lat_grid.flatten(), data.flatten(), function='linear')
    interpolated_data = rbf_interpolator(target_lon_grid, target_lat_grid)
    interpolated_variables[var] = interpolated_data

# 示例：组合地形数据和插值后的某个水体数据（例如温度）
combined_data = elevation * interpolated_variables['thetao']

# 可视化组合后的数据
plt.figure(figsize=(10, 8))
plt.imshow(combined_data, cmap='viridis', extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
plt.colorbar(label='Combined Value')
plt.title('Combined Terrain and Interpolated Water Data (Temperature)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# %% [cell 3]
import os
import rasterio
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

def interpolate_and_merge(terrain_file, water_file, output_dir):
    # 创建保存插值后数据的目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取地形文件
    with rasterio.open(terrain_file) as src:
        elevation = src.read(1)  # 读取地形数据
        bounds = src.bounds  # 获取地形边界
        transform = src.transform  # 获取仿射变换，用于定位经纬度
        resolution = src.res  # 获取分辨率
        # 计算地形数据的经纬度网格
        lons = np.linspace(bounds.left, bounds.right, src.width)
        lats = np.linspace(bounds.top, bounds.bottom, src.height)

    # 读取水体数据
    ds = xr.open_dataset(water_file)

    # 提取原始水体数据的纬度、经度
    original_lats = ds['latitude'].values
    original_lons = ds['longitude'].values

    # 创建网格，用于插值
    lon_grid, lat_grid = np.meshgrid(original_lons, original_lats)

    # 将地形的纬度和经度展开，生成插值目标网格
    target_lon_grid, target_lat_grid = np.meshgrid(lons, lats)

    # 对所有水体变量进行插值
    variables_to_interpolate = ['so', 'thetao', 'uo', 'vo', 'zos', 'utide', 'utotal', 'vtide', 'vtotal']
    interpolated_variables = {}

    for var in variables_to_interpolate:
        print(f"Interpolating variable: {var}")
        data = ds[var].isel(time=0, depth=0).values  # 选择特定时间和深度的数据

        # 检查并处理 inf 和 NaN
        valid_mask = np.isfinite(data)
        if not valid_mask.any():
            print(f"Warning: All values are NaN or Inf for variable {var}. Skipping.")
            continue

        # 提取有效的数据点进行插值
        valid_lon = lon_grid[valid_mask]
        valid_lat = lat_grid[valid_mask]
        valid_data = data[valid_mask]

        # 使用 RBF 插值，将原始水体数据插值到高分辨率网格上
        rbf_interpolator = Rbf(valid_lon, valid_lat, valid_data, function='linear')
        interpolated_data = rbf_interpolator(target_lon_grid, target_lat_grid)
        interpolated_variables[var] = interpolated_data

        # 保存插值后的数据到 NetCDF 文件
        interpolated_ds = xr.Dataset({var: (['latitude', 'longitude'], interpolated_data)},
                                     coords={'latitude': lats, 'longitude': lons})
        interpolated_ds.to_netcdf(os.path.join(output_dir, f"interpolated_{var}.nc"))

    # 将所有插值后的数据与地形数据合并
    combined_variables = {}
    for var, interpolated_data in interpolated_variables.items():
        combined_variables[var] = (['latitude', 'longitude'], elevation * interpolated_data)

    # 创建包含合并后数据的 Dataset
    combined_ds = xr.Dataset(combined_variables, coords={'latitude': lats, 'longitude': lons})

    # 保存合并后的数据到 NetCDF 文件
    combined_output_path = os.path.join(output_dir, "combined_with_terrain.nc")
    combined_ds.to_netcdf(combined_output_path)

    print(f"All interpolated and combined data have been saved to {output_dir}")

    # 输出合并后数据的一些信息
    print("Combined dataset:")
    print(combined_ds)
    
    print(f"Width (longitude dimension): {src.width}")
    print(f"Height (latitude dimension): {src.height}")


# 调用函数
interpolate_and_merge('./output/filtered_data.tif', './Data/GCPAF/combined_gcpaf_data.nc', './Data/Combined')

# %% [cell 4]
import os
import rasterio
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

def interpolate_and_merge(terrain_file, water_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Read terrain file
    with rasterio.open(terrain_file) as src:
        elevation = src.read(1)
        bounds = src.bounds
        lons = np.linspace(bounds.left, bounds.right, src.width)
        lats = np.linspace(bounds.top, bounds.bottom, src.height)

    # Read water data
    ds = xr.open_dataset(water_file)
    original_lats = ds['latitude'].values
    original_lons = ds['longitude'].values

    lon_grid, lat_grid = np.meshgrid(original_lons, original_lats)
    target_lon_grid, target_lat_grid = np.meshgrid(lons, lats)

    variables_to_interpolate = ['so', 'thetao', 'uo', 'vo', 'zos', 'utide', 'utotal', 'vtide', 'vtotal']
    interpolated_variables = {}

    for var in variables_to_interpolate:
        print(f"Interpolating variable: {var}")
        data = ds[var].isel(time=0, depth=0).values

        valid_mask = np.isfinite(data)
        if not valid_mask.any():
            print(f"Warning: All values are NaN or Inf for variable {var}. Skipping.")
            continue

        valid_lon = lon_grid[valid_mask]
        valid_lat = lat_grid[valid_mask]
        valid_data = data[valid_mask]

        rbf_interpolator = Rbf(valid_lon, valid_lat, valid_data, function='linear')
        interpolated_data = rbf_interpolator(target_lon_grid, target_lat_grid)
        interpolated_variables[var] = interpolated_data

        interpolated_ds = xr.Dataset({var: (['latitude', 'longitude'], interpolated_data)},
                                     coords={'latitude': lats, 'longitude': lons})
        interpolated_ds.to_netcdf(os.path.join(output_dir, f"interpolated_{var}.nc"))

    combined_variables = {}
    for var, interpolated_data in interpolated_variables.items():
        combined_variables[var] = (['latitude', 'longitude'], elevation * interpolated_data)

    combined_ds = xr.Dataset(combined_variables, coords={'latitude': lats, 'longitude': lons})
    combined_output_path = os.path.join(output_dir, "combined_with_terrain.nc")
    combined_ds.to_netcdf(combined_output_path)

    print(f"All interpolated and combined data have been saved to {output_dir}")
    print("Combined dataset:")
    print(combined_ds)

# Call the function
interpolate_and_merge('./output/filtered_data.tif', './Data/GCPAF/combined_gcpaf_data.nc', './Data/Combined')

# %% [cell 5]
def print_combined_data_structure(combined_file):
    # Read combined data file
    combined_ds = xr.open_dataset(combined_file)
    # Print the structure of the dataset
    print("Combined dataset structure:")
    print(combined_ds)



# Print the structure of the combined dataset
print_combined_data_structure('./Data/Combined/combined_with_terrain.nc')

# %% [cell 6]
import os
import rasterio
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

def interpolate_and_merge(terrain_file, water_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Read terrain file
    with rasterio.open(terrain_file) as src:
        elevation = src.read(1)
        bounds = src.bounds
        lons = np.linspace(bounds.left, bounds.right, src.width)
        lats = np.linspace(bounds.top, bounds.bottom, src.height)

    # Read water data
    ds = xr.open_dataset(water_file)
    original_lats = ds['latitude'].values
    original_lons = ds['longitude'].values

    lon_grid, lat_grid = np.meshgrid(original_lons, original_lats)
    target_lon_grid, target_lat_grid = np.meshgrid(lons, lats)

    variables_to_interpolate = ['so', 'thetao', 'uo', 'vo', 'zos', 'utide', 'utotal', 'vtide', 'vtotal']
    interpolated_variables = {}

    for var in variables_to_interpolate:
        print(f"Interpolating variable: {var}")
        data = ds[var].isel(time=0, depth=0).values

        valid_mask = np.isfinite(data)
        if not valid_mask.any():
            print(f"Warning: All values are NaN or Inf for variable {var}. Skipping.")
            continue

        valid_lon = lon_grid[valid_mask]
        valid_lat = lat_grid[valid_mask]
        valid_data = data[valid_mask]

        rbf_interpolator = Rbf(valid_lon, valid_lat, valid_data, function='linear')
        interpolated_data = rbf_interpolator(target_lon_grid, target_lat_grid)
        interpolated_variables[var] = interpolated_data

        interpolated_ds = xr.Dataset({var: (['latitude', 'longitude'], interpolated_data)},
                                     coords={'latitude': lats, 'longitude': lons})
        interpolated_ds.to_netcdf(os.path.join(output_dir, f"interpolated_{var}.nc"))

    combined_variables = {}
    for var, interpolated_data in interpolated_variables.items():
        combined_variables[var] = (['latitude', 'longitude'], elevation * interpolated_data)

    combined_ds = xr.Dataset(combined_variables, coords={'latitude': lats, 'longitude': lons})
    combined_output_path = os.path.join(output_dir, "combined_with_terrain.nc")
    combined_ds.to_netcdf(combined_output_path)

    print(f"All interpolated and combined data have been saved to {output_dir}")
    print("Combined dataset:")
    print(combined_ds)

def visualize_combined_data(combined_file):
    # Read combined data file
    combined_ds = xr.open_dataset(combined_file)

    # Plot each variable in the combined dataset
    for var in combined_ds.data_vars:
        plt.figure(figsize=(10, 6))
        combined_ds[var].plot()
        plt.title(f"Combined Data: {var}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()

# Call the function
interpolate_and_merge('./output/filtered_data.tif', './Data/GCPAF/combined_gcpaf_data.nc', './Data/Combined')

# Visualize the combined data
visualize_combined_data('./Data/Combined/combined_with_terrain.nc')

# %% [cell 7]
import os
import rasterio
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

def interpolate_and_merge(terrain_file, water_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Read terrain file
    with rasterio.open(terrain_file) as src:
        elevation = src.read(1)
        bounds = src.bounds
        lons = np.linspace(bounds.left, bounds.right, src.width)
        lats = np.linspace(bounds.top, bounds.bottom, src.height)

    # Read water data
    ds = xr.open_dataset(water_file)
    original_lats = ds['latitude'].values
    original_lons = ds['longitude'].values
    original_depths = ds['depth'].values
    original_times = ds['time'].values

    lon_grid, lat_grid = np.meshgrid(original_lons, original_lats)
    target_lon_grid, target_lat_grid = np.meshgrid(lons, lats)

    variables_to_interpolate = ['so', 'thetao', 'uo', 'vo', 'zos', 'utide', 'utotal', 'vtide', 'vtotal']
    interpolated_variables = {var: [] for var in variables_to_interpolate}

    for var in variables_to_interpolate:
        print(f"Interpolating variable: {var}")

        for t_idx, time in enumerate(original_times):
            for d_idx, depth in enumerate(original_depths):
                print(f"  Time: {time}, Depth: {depth}")
                data = ds[var].isel(time=t_idx, depth=d_idx).values

                valid_mask = np.isfinite(data)
                if not valid_mask.any():
                    print(f"  Warning: All values are NaN or Inf for variable {var} at time {time}, depth {depth}. Skipping.")
                    interpolated_variables[var].append(np.full((len(lats), len(lons)), np.nan))
                    continue

                valid_lon = lon_grid[valid_mask]
                valid_lat = lat_grid[valid_mask]
                valid_data = data[valid_mask]

                rbf_interpolator = Rbf(valid_lon, valid_lat, valid_data, function='linear')
                interpolated_data = rbf_interpolator(target_lon_grid, target_lat_grid)
                interpolated_variables[var].append(interpolated_data)

    combined_variables = {}
    for var, interpolated_list in interpolated_variables.items():
        # Stack the list of interpolated data along time and depth dimensions
        stacked_data = np.stack(interpolated_list, axis=0)
        stacked_data = stacked_data.reshape((len(original_times), len(original_depths), len(lats), len(lons)))
        combined_variables[var] = (['time', 'depth', 'latitude', 'longitude'], stacked_data)

    combined_ds = xr.Dataset(combined_variables, coords={'time': original_times, 'depth': original_depths, 'latitude': lats, 'longitude': lons})
    combined_output_path = os.path.join(output_dir, "combined_with_terrain.nc")
    combined_ds.to_netcdf(combined_output_path)

    print(f"All interpolated and combined data have been saved to {output_dir}")
    print("Combined dataset:")
    print(combined_ds)

def visualize_combined_data(combined_file, selected_time=None, selected_depth=None, time_index=None, depth_index=None):
    # Read combined data file
    combined_ds = xr.open_dataset(combined_file)

    # Determine selected time and depth
    if time_index is not None:
        selected_time = combined_ds['time'].values[time_index]
    if depth_index is not None:
        selected_depth = combined_ds['depth'].values[depth_index]

    # Output data characteristics for each variable
    for var in combined_ds.data_vars:
        data = combined_ds[var]
        print(f"Data characteristics for variable {var}:")
        print(f"  Mean: {data.mean().item()} ")
        print(f"  Std Dev: {data.std().item()} ")
        print(f"  Min: {data.min().item()} ")
        print(f"  Max: {data.max().item()} ")

    # Plot each variable in the combined dataset with different colormaps
    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'cool', 'hot', 'spring', 'summer']
    for i, var in enumerate(combined_ds.data_vars):
        if selected_time is not None and selected_depth is not None:
            plt.figure(figsize=(10, 6))
            combined_ds[var].sel(time=selected_time, depth=selected_depth).plot(cmap=colormaps[i % len(colormaps)])
            plt.title(f"Combined Data: {var} (Time: {selected_time}, Depth: {selected_depth})")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.show()
        else:
            for t_idx in range(len(combined_ds['time'])):
                for d_idx in range(len(combined_ds['depth'])):
                    plt.figure(figsize=(10, 6))
                    combined_ds[var].isel(time=t_idx, depth=d_idx).plot(cmap=colormaps[i % len(colormaps)])
                    plt.title(f"Combined Data: {var} (Time: {combined_ds['time'].values[t_idx]}, Depth: {combined_ds['depth'].values[d_idx]})")
                    plt.xlabel("Longitude")
                    plt.ylabel("Latitude")
                    plt.show()

def print_combined_data_structure(combined_file):
    combined_ds = xr.open_dataset(combined_file)
    print("Combined dataset structure:")
    print(combined_ds)

interpolate_and_merge('./output/filtered_data.tif', './Data/GCPAF/combined_gcpaf_data.nc', './Data/Combined')

# visualize
visualize_combined_data('./Data/Combined/combined_with_terrain.nc', time_index=0, depth_index=0)

# %% [cell 8]
import xarray as xr
import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_by_time(data_file, output_dir):
    # Read combined data file
    combined_ds = xr.open_dataset(data_file)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get time values
    times = combined_ds['time'].values

    # Loop through each variable and visualize its trend over time
    for var in combined_ds.data_vars:
        plt.figure(figsize=(12, 6))
        mean_values = []

        # Calculate the mean value for each time step
        for t_idx in range(len(times)):
            data = combined_ds[var].isel(time=t_idx)
            mean_values.append(data.mean().item())

        # Plot the trend over time
        plt.plot(times, mean_values, marker='o', linestyle='-', label=var)
        plt.title(f"Temporal Trend of {var}")
        plt.xlabel("Time")
        plt.ylabel(f"Mean {var}")
        plt.grid(True)
        plt.legend()

        # Save the figure
        output_path = os.path.join(output_dir, f"{var}_trend_over_time.png")
        plt.savefig(output_path)
        plt.close()

        print(f"Saved temporal trend visualization for variable '{var}' to '{output_path}'")

# Example usage
visualize_by_time('./Data/Combined/combined_with_terrain.nc', './Data/Visualizations')

# %% [cell 9]
import xarray as xr
import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_by_time(data_file, output_dir, lat_idx, lon_idx):
    # Read combined data file
    combined_ds = xr.open_dataset(data_file)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get time values
    times = combined_ds['time'].values

    # Loop through each variable and visualize its trend over time at a specific point
    for var in combined_ds.data_vars:
        plt.figure(figsize=(12, 6))
        point_values = []

        # Extract the value for each time step at the specified latitude and longitude index
        for t_idx in range(len(times)):
            data = combined_ds[var].isel(time=t_idx, latitude=lat_idx, longitude=lon_idx)
            point_values.append(data.item())

        # Plot the trend over time
        plt.plot(times, point_values, marker='o', linestyle='-', label=var)
        plt.title(f"Temporal Trend of {var} at Point (lat_idx={lat_idx}, lon_idx={lon_idx})")
        plt.xlabel("Time")
        plt.ylabel(f"{var} Value at Specified Point")
        plt.grid(True)
        plt.legend()

        # Save the figure
        output_path = os.path.join(output_dir, f"{var}_trend_over_time_at_point.png")
        plt.savefig(output_path)
        plt.close()

        print(f"Saved temporal trend visualization for variable '{var}' at point (lat_idx={lat_idx}, lon_idx={lon_idx}) to '{output_path}'")

# Example usage
visualize_by_time('./Data/Combined/combined_with_terrain.nc', './Data/Visualizations', lat_idx=10, lon_idx=15)

