import argparse


def parse_config():
    parser = argparse.ArgumentParser()
    # inference
    parser.add_argument('--input_H_W', type=int, default=416, help="416 or 608")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--anchors_path', type=str, default='model_data/yolo_anchors.txt')
    parser.add_argument('--classes_path', type=str, default='model_data/coco_classes.txt')
    parser.add_argument('--save_model_path', type=str, default='model_weight', help="saving of model's path")
    parser.add_argument('--weight_path', type=str, default='model_weight/model_weights.pth')
    parser.add_argument('--confidence', type=float, default=0.65, help="Object confidence threshold")
    parser.add_argument('--nms_thres', type=float, default=0.3)

    args = parser.parse_args()

    return args
