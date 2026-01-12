#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
import time
from datetime import datetime

# 定义源目录路径
SOURCE_DIR = "/mnt/data1/zdy/rl-spbs-summerV1_gitee/logs"


def main():
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="根据匹配字段复制日志文件夹到 /tmp")

    # 第一个参数：目标文件夹的前缀，默认为 'logs'
    parser.add_argument(
        "prefix",
        nargs='?',
        default="logs",
        help="目标文件夹的前缀名称 (默认为 logs)"
    )

    # 第二个参数：匹配字段，默认为 '*'
    parser.add_argument(
        "pattern",
        nargs='?',
        default="*",
        help="匹配文件夹名称的字段 (默认为 *，即复制所有)"
    )

    args = parser.parse_args()

    # 2. 创建目标文件夹
    # 获取当前时间，格式如 2026-01-12-13-45-17
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # 拼接目标文件夹名称
    dest_folder_name = f"{args.prefix}_{current_time}"
    dest_path = os.path.join("/tmp", dest_folder_name)

    try:
        os.makedirs(dest_path, exist_ok=True)
        print(f"[Info] 目标文件夹已创建: {dest_path}")
    except OSError as e:
        print(f"[Error] 创建文件夹失败: {e}")
        return

    # 3. 检查源目录是否存在
    if not os.path.exists(SOURCE_DIR):
        print(f"[Error] 源目录不存在: {SOURCE_DIR}")
        return

    # 4. 遍历源目录并复制匹配的文件夹
    print(f"[Info] 正在从 {SOURCE_DIR} 筛选...")
    print(f"[Info] 匹配模式: {'所有文件夹' if args.pattern == '*' else args.pattern}")

    count = 0
    for item in os.listdir(SOURCE_DIR):
        src_item_path = os.path.join(SOURCE_DIR, item)

        # 确保是文件夹才处理
        if os.path.isdir(src_item_path):
            is_match = False

            # 判断匹配逻辑
            if args.pattern == "*":
                is_match = True
            elif args.pattern in item:
                is_match = True

            if is_match:
                # 拼接目标子路径
                dst_item_path = os.path.join(dest_path, item)
                try:
                    # 使用 copytree 递归复制文件夹
                    shutil.copytree(src_item_path, dst_item_path)
                    print(f"  -> 已复制: {item}")
                    count += 1
                except Exception as e:
                    print(f"  [Error] 复制 {item} 失败: {e}")

    print(f"-" * 30)
    print(f"[Success] 任务完成。共复制了 {count} 个文件夹到 {dest_path}")


if __name__ == "__main__":
    main()
