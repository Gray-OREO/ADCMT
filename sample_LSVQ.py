import pandas as pd
import numpy as np
import os
import shutil


def stratified_sample_csv(input_path, output_path, score_column='score', columns_to_save=None, bins=10):
    """
    读取CSV文件，按分数分布均匀抽取1/10的条目，并保存指定列

    参数:
        input_path (str): 输入CSV文件路径
        output_path (str): 输出CSV文件路径
        score_column (str): 分数列名 (默认'score')
        columns_to_save (list): 需要保存的列名 (默认保存所有列)
        bins (int): 分层区间数 (默认分10个区间)
    """
    # 读取CSV文件
    df = pd.read_csv(input_path)

    # 检查分数列是否存在
    if score_column not in df.columns:
        raise ValueError(f"分数列 '{score_column}' 不存在")

    # 创建分数分层
    df['strata'] = pd.cut(df[score_column], bins=bins, labels=False)

    # 按分层采样
    sampled = df.groupby('strata', group_keys=False).apply(
        lambda x: x.sample(frac=0.1, random_state=19980427)
    )

    # 选择需要保存的列
    if columns_to_save:
        sampled = sampled[columns_to_save]

    # 保存结果
    sampled.to_csv(output_path, index=False)
    print(f"已保存筛选后的数据到 {output_path}，共 {len(sampled)} 条记录")


def sync_csv_with_directory(
        original_csv: str,
        corrected_csv: str,
        target_dir: str,
        name_column: str = "name",
        path_separator: str = "/",
        file_extension: str = None
):
    """
    根据目标目录实际存在的文件同步修正CSV

    参数:
        original_csv: 原始CSV文件路径
        corrected_csv: 修正后的CSV保存路径
        target_dir: 目标目录路径
        name_column: 包含文件名的列名 (默认'name')
        path_separator: 路径分隔符 (默认'/')
        file_extension: 目标文件扩展名 (如'.mp4'，默认None表示保留原始扩展名)
    """
    try:
        # 读取原始CSV
        df = pd.read_csv(original_csv)

        # 验证必要列存在
        if name_column not in df.columns:
            raise ValueError(f"CSV中不存在指定的文件名列: {name_column}")

        # 提取文件名
        def extract_filename(name_str):
            # 分割路径获取基础文件名
            base_name = name_str.split(path_separator)[-1]
            # 如果需要强制修改扩展名
            if file_extension:
                return os.path.splitext(base_name)[0] + file_extension
            return base_name

        df['_sync_filename'] = df[name_column].apply(extract_filename)

        # 获取目标目录实际文件列表
        existing_files = set(os.listdir(target_dir))

        # 过滤数据
        filtered_df = df[df['_sync_filename'].isin(existing_files)].copy()
        filtered_df.drop('_sync_filename', axis=1, inplace=True)

        # 保存修正后的CSV
        filtered_df.to_csv(corrected_csv, index=False)

        print(f"同步完成，原始记录数: {len(df)}，有效记录数: {len(filtered_df)}")
        print(f"缺失文件数: {len(df) - len(filtered_df)}")
        print(f"修正后的CSV已保存至: {corrected_csv}")

    except FileNotFoundError:
        print(f"错误：文件不存在 {original_csv}")
    except pd.errors.EmptyDataError:
        print("错误：CSV文件内容为空")
    except Exception as e:
        print(f"处理失败: {str(e)}")


def clean_extra_videos(
        csv_path: str,
        target_folder: str,
        name_column: str = "name",
        path_separator: str = "/",
        allowed_extensions: list = [".mp4", ".avi", ".mov"],
        backup_folder: str = None,
        dry_run: bool = False
):
    """
    根据CSV清理目标文件夹中的多余视频文件

    参数:
        csv_path: 修正后的CSV文件路径
        target_folder: 要清理的目标文件夹路径
        name_column: CSV中文件名的列名 (默认'name')
        path_separator: 文件名中的路径分隔符 (默认'/')
        allowed_extensions: 允许的视频扩展名列表
        backup_folder: 备份目录路径 (默认不备份)
        dry_run: 试运行模式 (仅显示不实际删除)
    """
    try:
        # 读取CSV获取有效文件名列表
        df = pd.read_csv(csv_path)
        valid_files = set()

        # 处理文件名
        for name in df[name_column]:
            # 提取基础文件名
            base_name = name.split(path_separator)[-1]
            # 生成所有可能的扩展名组合
            for ext in allowed_extensions:
                valid_files.add(base_name)
                valid_files.add(f"{base_name}{ext}")
                valid_files.add(os.path.splitext(base_name)[0] + ext)

        # 获取目标目录实际文件列表
        existing_files = set(os.listdir(target_folder))

        # 计算需要删除的文件列表
        to_delete = [f for f in existing_files if f not in valid_files]

        # 过滤非视频文件
        to_delete = [f for f in to_delete if os.path.splitext(f)[1].lower() in allowed_extensions]

        # 备份处理
        if backup_folder and not dry_run:
            os.makedirs(backup_folder, exist_ok=True)
            for f in to_delete:
                src = os.path.join(target_folder, f)
                dst = os.path.join(backup_folder, f)
                shutil.move(src, dst)
            print(f"已备份 {len(to_delete)} 个文件到 {backup_folder}")
        elif not dry_run:
            # 直接删除
            for f in to_delete:
                os.remove(os.path.join(target_folder, f))

        # 输出报告
        print(f"目标目录文件总数: {len(existing_files)}")
        print(f"有效文件数: {len(valid_files)}")
        print(f"待删除文件数: {len(to_delete)}")
        if dry_run:
            print("[试运行模式] 以下文件将被删除:", "\n".join(to_delete))
        else:
            print(f"已删除 {len(to_delete)} 个文件")

    except FileNotFoundError as e:
        print(f"文件不存在: {str(e)}")
    except pd.errors.EmptyDataError:
        print("CSV文件内容为空")
    except PermissionError:
        print("权限不足，请以管理员身份运行")
    except Exception as e:
        print(f"处理失败: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # ========================= Sample ================================
    # input_csv = "G:/Database/LSVQ/labels_train_test.csv"  # 输入文件路径
    # output_csv = "LSVQs_data.csv"  # 输出文件路径
    #
    # # 指定需要保存的列（若不需要筛选列，设为None）
    # selected_columns = ['name', 'height', 'width', 'frame_number', 'mos']
    #
    # stratified_sample_csv(
    #     input_path=input_csv,
    #     output_path=output_csv,
    #     score_column='mos',
    #     columns_to_save=selected_columns,
    #     bins=50
    # )

    # ============================= Extract ============================
    from tqdm import tqdm

    input_csv = "LSVQs_data.csv"  # 输入文件路径
    df = pd.read_csv(input_csv)
    root = 'G:/Database/'

    for f_name in tqdm(df['name'], total=len(df)):
        src_path = os.path.join(root, 'LSVQ', f_name + '.mp4')  # 构建源文件路径
        # f_name_ = f_name.split('/')[1]  # 提取文件名（假设原路径中有子目录）
        dst_path = os.path.join(root, 'LSVQs', 'videos', f_name + '.mp4')  # 构建目标路径

        # 检查目标文件是否存在
        if os.path.exists(dst_path):
            print(f"视频文件 {f_name} 已存在，跳过复制。")
            continue  # 跳过当前循环的复制操作

        # 确保目标目录存在（可选，根据需要添加）
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # 执行文件复制
        shutil.copy2(src_path, dst_path)
    # ============================ Clean ======================================
    # input_csv = "LSVQs_data.csv"
    # output_csv = "LSVQs_metadata.csv"
    # target_dir = "G:/Database/LSVQs/videos"  # 替换为实际目标文件夹路径
    #
    # # 执行处理
    # sync_csv_with_directory(
    #     original_csv=input_csv,
    #     corrected_csv=output_csv,
    #     target_dir=target_dir,
    #     name_column="name",
    #     path_separator="/",
    #     file_extension=".mp4"
    # )
    # =======================================================
    # clean_extra_videos(
    #     csv_path="LSVQs_metadata.csv",
    #     target_folder="G:/Database/LSVQs/videos",
    #     name_column="name",
    #     path_separator="/",
    #     allowed_extensions=[".mp4"],
    #     backup_folder="G:/Database/LSVQs/backup",
    #     dry_run=False
    # )