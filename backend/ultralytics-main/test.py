from ultralytics import YOLO
import os
import torch
import argparse
import cv2
import numpy as np
from pathlib import Path


def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='YOLOv8 测试脚本')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径 (.pt)')
    parser.add_argument('--data', type=str, default=None, help='数据配置文件路径 (yaml)')
    parser.add_argument('--source', type=str, default=None, help='测试图像或视频路径')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU阈值')
    parser.add_argument('--device', type=str, default='0', help='设备 (例如: 0, 0,1,2,3, cpu)')
    parser.add_argument('--save-txt', action='store_true', help='保存结果为txt文件')
    parser.add_argument('--save-conf', action='store_true', help='在txt文件中保存置信度')
    parser.add_argument('--save-crop', action='store_true', help='裁剪检测框')
    parser.add_argument('--save-dir', type=str, default='runs/detect', help='保存结果的目录')
    parser.add_argument('--view-img', action='store_true', help='显示结果')
    parser.add_argument('--hide-labels', action='store_true', help='隐藏标签')
    parser.add_argument('--hide-conf', action='store_true', help='隐藏置信度')
    parser.add_argument('--mode', type=str, default='predict', choices=['predict', 'val'],
                        help='执行模式: predict(预测)或val(验证)')

    args = parser.parse_args()

    # 打印系统和 CUDA 信息
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备数量: {torch.cuda.device_count()}")
        print(f"CUDA 设备名称: {torch.cuda.get_device_name(0)}")

    # 设置 CUDA 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"错误: 模型文件 '{args.model}' 不存在!")
        return

    # 如果是验证模式, 检查data文件是否存在
    if args.mode == 'val' and (not args.data or not os.path.exists(args.data)):
        print(f"错误: 验证模式需要有效的数据配置文件!")
        return

    # 如果是预测模式, 检查source是否存在
    if args.mode == 'predict' and not args.source:
        print(f"错误: 预测模式需要指定source参数!")
        return

    try:
        # 加载模型
        model = YOLO(args.model)
        print(f"成功加载模型: {args.model}")

        # 根据模式执行相应操作
        if args.mode == 'val':
            print(f"\n开始验证数据集: {args.data}")

            # 验证设置
            val_args = {
                'data': args.data,
                'imgsz': args.imgsz,
                'batch': 1,  # 验证时通常使用较小的batch
                'device': args.device,
                'conf': args.conf,
                'iou': args.iou,
                'verbose': True
            }

            # 执行验证
            val_results = model.val(**val_args)

            # 打印验证结果
            print("\n验证结果:")
            print(f"mAP50-95: {val_results.box.map}")
            print(f"mAP50: {val_results.box.map50}")
            print(f"mAP75: {val_results.box.map75}")
            print(f"精确度 (Precision): {val_results.box.p}")
            print(f"召回率 (Recall): {val_results.box.r}")

            # 打印每个类别的结果
            if hasattr(val_results.box, 'cls') and val_results.box.cls is not None:
                print("\n各类别的表现:")
                for i, cls_name in enumerate(val_results.names):
                    idx = val_results.box.cls == i
                    if idx.sum() > 0:
                        print(f"类别 '{cls_name}':")
                        print(f"  样本数: {idx.sum()}")
                        print(f"  精确度: {val_results.box.p[idx].mean():.4f}")
                        print(f"  召回率: {val_results.box.r[idx].mean():.4f}")

        else:  # predict 模式
            print(f"\n开始预测: {args.source}")

            # 创建保存目录
            os.makedirs(args.save_dir, exist_ok=True)

            # 预测设置
            predict_args = {
                'source': args.source,
                'imgsz': args.imgsz,
                'conf': args.conf,
                'iou': args.iou,
                'save': True,  # 保存结果
                'save_txt': args.save_txt,
                'save_conf': args.save_conf,
                'save_crop': args.save_crop,
                'show': args.view_img,
                'device': args.device,
                'project': args.save_dir,
                'name': 'exp',
                'hide_labels': args.hide_labels,
                'hide_conf': args.hide_conf,
                'verbose': True
            }

            # 执行预测
            results = model.predict(**predict_args)

            # 打印预测结果摘要
            print("\n预测结果摘要:")
            total_objects = sum(len(r.boxes) for r in results)
            print(f"总检测到的对象数量: {total_objects}")

            # 如果结果中包含多个类别，打印每个类别的检测数量
            if len(results) > 0 and hasattr(results[0], 'names'):
                class_counts = {}
                for r in results:
                    if r.boxes.cls.numel() > 0:
                        for cls_id in r.boxes.cls.cpu().numpy():
                            cls_name = r.names[int(cls_id)]
                            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                print("\n各类别检测数量:")
                for cls_name, count in class_counts.items():
                    print(f"  {cls_name}: {count}")

            print(f"\n结果已保存至: {os.path.join(args.save_dir, 'exp')}")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")


if __name__ == "__main__":
    main()
    #python test.py --model G:\yolo_learn\ultralytics-main\ultralytics-main\runs\train\exp\weights\best.pt --mode val --data G:\yolo_learn\ultralytics-main\ultralytics-main\wheat.yaml