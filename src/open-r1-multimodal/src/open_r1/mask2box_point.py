import os
import numpy as np
import cv2



def process_mask(
    img,img_path,
    out_image_path: str = None,
    bin_threshold: int = 0
):
    """
    Parameters
    ----------
    mask_path : str
        输入的mask图路径。支持灰度或彩色，非零像素视为前景。
    out_image_path : str
        标注结果保存路径；若为空则保存为 mask_path 同目录下的 *_vis.png
    out_json_path : str
        输出坐标信息保存路径；若为空则保存为 *_info.json
    bin_threshold : int
        二值化阈值（0 表示自动按非零作为前景；>0 则先转灰度后用该阈值二值化）
    """
    # 1) 读图
    assert len(img.shape)==2
    h, w = img.shape[:2]

    # 1) 转二值图：非零为前景（如果提供阈值则用阈值）
    if bin_threshold > 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        _, bin_mask = cv2.threshold(gray, bin_threshold, 255, cv2.THRESH_BINARY)
    else:
        if img.ndim == 3:
            # 任一通道非零视为前景
            bin_mask = (np.any(img != 0, axis=2)).astype(np.uint8) * 255
        else:
            bin_mask = (img != 0).astype(np.uint8) * 255

    # 如果没有前景，直接返回空结果
    if cv2.countNonZero(bin_mask) == 0:
        info = {
            "image_size": [w, h],
            "global_box": None,
            "instances": [],
            "note": "No foreground found in mask."
        }
        # base, _ = os.path.splitext(mask_path)

        out_image_path = f"{out_image_path}/{os.path.basename(img_path)}_vis.png"
        # 直接存原图（或白底）并写json
        vis = cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(out_image_path, vis)
        return []#,[]

    # 2) 计算整体 box（包含所有前景像素）
    ys, xs = np.where(bin_mask > 0)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    global_box = [int(x1), int(y1), int(x2), int(y2)]
    return global_box


