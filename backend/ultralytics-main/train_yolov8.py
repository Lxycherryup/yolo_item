from ultralytics import YOLO
import os
import torch
import argparse


def main():
    # 创建参数解析器，以便稍后可以自定义参数
    parser = argparse.ArgumentParser(description='YOLOv8 训练脚本')
    parser.add_argument('--data', type=str, default='wheat.yaml', help='数据配置文件路径')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='模型文件路径')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--batch', type=int, default=24, help='批处理大小')
    parser.add_argument('--workers', type=int, default=0, help='数据加载的工作线程数')
    parser.add_argument('--device', type=str, default='0', help='训练设备 (例如: 0, 0,1,2,3, cpu)')
    parser.add_argument('--project', type=str, default='runs/train', help='保存结果的项目目录')
    parser.add_argument('--name', type=str, default='exp_batch16', help='保存结果的名称')
    parser.add_argument('--patience', type=int, default=50, help='EarlyStopping 的耐心值')
    parser.add_argument('--save-period', type=int, default=-1, help='每 x 个 epoch 保存一次模型')
    parser.add_argument('--exist-ok', action='store_true', help='存在时覆盖现有实验')
    parser.add_argument('--pretrained', action='store_true', default=True, help='使用预训练模型')
    parser.add_argument('--optimizer', type=str, default='auto', help='优化器 (auto, SGD, Adam, AdamW, RMSProp)')
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='权重衰减')
    parser.add_argument('--dropout', type=float, default=0.0, help='使用 dropout，值介于0到1之间')
    parser.add_argument('--amp', action='store_true', help='是否使用自动混合精度训练')

    args = parser.parse_args()

    # 打印系统和 CUDA 信息
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备数量: {torch.cuda.device_count()}")
        print(f"CUDA 设备名称: {torch.cuda.get_device_name(0)}")

    # 设置 CUDA 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 加载模型
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    # 训练设置
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'workers': args.workers,
        'device': args.device,
        'project': args.project,
        'name': args.name,
        'patience': args.patience,
        'save_period': args.save_period,
        'exist_ok': args.exist_ok,
        'pretrained': args.pretrained,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'verbose': True,  # 显示详细训练信息
        'seed': 42,  # 设置随机种子以便结果可复现
        'amp': False,  # 禁用自动混合精度训练
    }

    # 打印训练配置
    print("\n训练配置:")
    for key, value in train_args.items():
        print(f"{key}: {value}")
    print("\n开始训练...\n")

    # 计算最佳模型路径
    best_model_path = os.path.join(args.project, args.name, "weights", "best.pt")

    # 开始训练
    try:
        results = model.train(**train_args)

        # 打印训练完成后的结果
        print("\n训练完成！")
        # 修正: 使用预计算的最佳模型路径
        print(f"最佳模型保存在: {best_model_path}")

        # 显示训练结果指标
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print("\n训练结果摘要:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.6f}")
                else:
                    print(f"{key}: {value}")
        else:
            print("无法获取训练结果指标")

        # 在验证集上验证最终模型
        print("\n在验证集上验证最终模型...")
        try:
            # 加载最佳模型进行验证
            best_model = YOLO(best_model_path)
            val_results = best_model.val(data=args.data)
            print("验证结果:")
            print(f"mAP50-95: {val_results.box.map}")
            print(f"mAP50: {val_results.box.map50}")
            print(f"mAP75: {val_results.box.map75}")
        except Exception as e:
            print(f"验证过程中发生错误: {e}")

    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        if os.path.exists(best_model_path):
            print(f"尽管训练过程中发生了错误，但最佳模型可能已保存在: {best_model_path}")


if __name__ == "__main__":
    main()