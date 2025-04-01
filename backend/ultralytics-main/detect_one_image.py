from ultralytics import YOLO
import os
import torch
import cv2
import numpy as np
from pathlib import Path


def detect_single_image(model, image_path, imgsz=640, conf=0.25, iou=0.45, save_dir='runs/detect/single',
                        hide_labels=False, hide_conf=False, save_txt=False, save_conf=False, save_crop=False,
                        view_img=True):
    """
    检测单张图片并显示结果

    Args:
        model: 加载的YOLO模型
        image_path: 图片路径
        imgsz: 图像尺寸
        conf: 置信度阈值
        iou: IoU阈值
        save_dir: 保存结果的目录
        hide_labels: 是否隐藏标签
        hide_conf: 是否隐藏置信度
        save_txt: 是否保存结果为txt文件
        save_conf: 是否在txt文件中保存置信度
        save_crop: 是否裁剪检测框
        view_img: 是否显示结果图像

    Returns:
        处理后的图像
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图片 '{image_path}' 不存在!")
        return None

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图片 '{image_path}'!")
        return None

    # 获取图片原始尺寸
    orig_height, orig_width = img.shape[:2]

    # 执行预测
    results = model.predict(
        source=image_path,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        save=True,
        save_txt=save_txt,
        save_conf=save_conf,
        save_crop=save_crop,
        project=save_dir,
        name='exp',
        hide_labels=hide_labels,
        hide_conf=hide_conf,
        verbose=False
    )[0]  # 只取第一个结果，因为只有一张图片

    # 获取结果图像路径
    result_path = os.path.join(save_dir, 'exp', os.path.basename(image_path))

    # 读取处理后的图像
    processed_img = cv2.imread(result_path) if os.path.exists(result_path) else None

    # 打印检测结果
    print("\n检测结果:")
    if len(results.boxes) > 0:
        print(f"检测到 {len(results.boxes)} 个目标")

        # 统计每个类别的数量
        class_counts = {}
        for i in range(len(results.boxes)):
            box = results.boxes[i]
            cls_id = int(box.cls.item())
            cls_name = results.names[cls_id]
            conf_val = box.conf.item()

            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            # 打印每个检测框的信息
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            print(f"目标 {i + 1}: 类别='{cls_name}', 置信度={conf_val:.4f}, 坐标=({x1}, {y1}, {x2}, {y2})")

        # 打印每个类别的数量
        print("\n各类别检测数量:")
        for cls_name, count in class_counts.items():
            print(f"  {cls_name}: {count}")
    else:
        print("未检测到任何目标")

    print(f"\n结果已保存至: {os.path.join(save_dir, 'exp')}")

    # 如果需要显示图片并且处理后的图片存在
    if view_img and processed_img is not None:
        # 显示图片
        cv2.namedWindow('YOLOv8 检测结果', cv2.WINDOW_NORMAL)
        cv2.imshow('YOLOv8 检测结果', processed_img)
        print("按任意键关闭图片窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return processed_img


def validate_model(model, data_yaml, imgsz=640, conf=0.25, iou=0.45, device='0'):
    """
    验证模型在数据集上的性能

    Args:
        model: 加载的YOLO模型
        data_yaml: 数据集配置文件
        imgsz: 图像尺寸
        conf: 置信度阈值
        iou: IoU阈值
        device: 计算设备
    """
    print(f"\n开始验证数据集: {data_yaml}")

    # 验证设置
    val_args = {
        'data': data_yaml,
        'imgsz': imgsz,
        'batch': 1,  # 验证时通常使用较小的batch
        'device': device,
        'conf': conf,
        'iou': iou,
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


def predict_on_images(model, source_path, imgsz=640, conf=0.25, iou=0.45, save_dir='runs/detect',
                      hide_labels=False, hide_conf=False, save_txt=False, save_conf=False,
                      save_crop=False, view_img=False, device='0'):
    """
    对一组图像或视频进行预测

    Args:
        model: 加载的YOLO模型
        source_path: 源图像或视频路径
        imgsz: 图像尺寸
        conf: 置信度阈值
        iou: IoU阈值
        save_dir: 保存结果的目录
        hide_labels: 是否隐藏标签
        hide_conf: 是否隐藏置信度
        save_txt: 是否保存结果为txt文件
        save_conf: 是否在txt文件中保存置信度
        save_crop: 是否裁剪检测框
        view_img: 是否显示结果
        device: 计算设备
    """
    print(f"\n开始预测: {source_path}")

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 预测设置
    predict_args = {
        'source': source_path,
        'imgsz': imgsz,
        'conf': conf,
        'iou': iou,
        'save': True,  # 保存结果
        'save_txt': save_txt,
        'save_conf': save_conf,
        'save_crop': save_crop,
        'show': view_img,
        'device': device,
        'project': save_dir,
        'name': 'exp',
        'hide_labels': hide_labels,
        'hide_conf': hide_conf,
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

    print(f"\n结果已保存至: {os.path.join(save_dir, 'exp')}")


def main():
    """
    主函数，在这里设置所有参数
    """
    # === 配置参数（直接在代码中修改这些参数）===
    # 模型配置
    model_path = r"G:\yolo_learn\ultralytics-main\ultralytics-main\runs\train\exp_batch16\weights\best.pt"  # 修改为你的模型路径
    device = "0"  # 使用的GPU设备，使用CPU则设为"cpu"

    # 检测配置
    imgsz = 640  # 图像尺寸
    conf = 0.25  # 置信度阈值
    iou = 0.45  # IoU阈值

    # 显示和保存配置
    save_txt = False  # 是否保存结果为txt文件
    save_conf = False  # 是否在txt文件中保存置信度
    save_crop = False  # 是否裁剪检测框
    save_dir = "runs/detect"  # 保存结果的目录
    view_img = True  # 是否显示结果
    hide_labels = False  # 是否隐藏标签
    hide_conf = False  # 是否隐藏置信度

    # 运行模式配置
    mode = "single"  # 运行模式: "single"(单张图片) / "predict"(预测一组图像) / "val"(验证)

    # 特定模式的额外参数
    single_img_path = r"G:\yolo_learn\ultralytics-main\ultralytics-main\data\Images\BloodImage_00000.jpg"  # 单张图片路径（single模式）
    source_path = "path/to/images"  # 预测的图像或视频路径（predict模式）
    data_yaml = "path/to/data.yaml"  # 数据集配置文件（val模式）

    # === 系统信息 ===
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备数量: {torch.cuda.device_count()}")
        print(f"CUDA 设备名称: {torch.cuda.get_device_name(0)}")

    # 设置 CUDA 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    # === 参数验证 ===
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 不存在!")
        return

    # 根据模式检查必要的参数
    if mode == 'single' and not os.path.exists(single_img_path):
        print(f"错误: 单张图片 '{single_img_path}' 不存在!")
        return
    elif mode == 'val' and (not data_yaml or not os.path.exists(data_yaml)):
        print(f"错误: 验证模式需要有效的数据配置文件!")
        return
    elif mode == 'predict' and not os.path.exists(source_path):
        print(f"错误: 源路径 '{source_path}' 不存在!")
        return

    try:
        # 加载模型
        model = YOLO(model_path)
        print(f"成功加载模型: {model_path}")

        # 根据模式执行相应操作
        if mode == 'single':
            print(f"\n开始检测单张图片: {single_img_path}")
            # 执行单张图片检测
            detect_single_image(
                model=model,
                image_path=single_img_path,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                save_dir=save_dir,
                hide_labels=hide_labels,
                hide_conf=hide_conf,
                save_txt=save_txt,
                save_conf=save_conf,
                save_crop=save_crop,
                view_img=view_img
            )

        elif mode == 'val':
            # 执行验证
            validate_model(
                model=model,
                data_yaml=data_yaml,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device
            )

        else:  # predict 模式
            # 执行预测
            predict_on_images(
                model=model,
                source_path=source_path,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                save_dir=save_dir,
                hide_labels=hide_labels,
                hide_conf=hide_conf,
                save_txt=save_txt,
                save_conf=save_conf,
                save_crop=save_crop,
                view_img=view_img,
                device=device
            )

    except Exception as e:
        print(f"测试过程中发生错误: {e}")


if __name__ == "__main__":
    main()