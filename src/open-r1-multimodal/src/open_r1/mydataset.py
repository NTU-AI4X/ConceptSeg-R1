from open_r1.mask2box_point import process_mask
import torchvision.transforms.functional as TF
from PIL import Image as PILImage
from PIL import ImageDraw
import tqdm
import math,json
import torch
from open_r1.constants import *
import random
def get_cd_data(data_files, image_folders,mode,dataset_name_list):
    if isinstance(dataset_name_list, str):
        dataset_name_list = [dataset_name_list]
        
    all_data = []
    for data_file, image_folder in zip(data_files, image_folders):
        with open(data_file, "r") as f:
            data = json.load(f)
        id = 1
        for context_datasets in tqdm.tqdm(data[mode].values(), desc="processing context_datasets"):

            for item in context_datasets:
                if item["dataset_name"] not in  dataset_name_list:continue

                if len(item["img_path"])==1:
                    ref_cls_name = item["cls_name"]#.replace("_defect", "_good")
                    ref_images,ref_masks = [],[]
                    sample_num = 1 if len(data["train"][ref_cls_name])==2 else 4
                    for _ in range(sample_num):
                        ref_item = random.choice(data["train"][ref_cls_name])
                        while ref_item["img_path"][0] == item["img_path"][0]:
                            ref_item = random.choice(data["train"][ref_cls_name])
                        ref_images.append(os.path.join(image_folder, ref_item["img_path"][0]))
                        ref_masks.append(os.path.join(image_folder, ref_item["mask_path"]))
                    mask_path =  [*ref_masks,os.path.join(image_folder, item["mask_path"])]
                    target_img = os.path.join(image_folder, item["img_path"][0])
                else:
                    ref_images, ref_masks = [], None
                    for p in item["img_path"][:-1]:
                        ref_images.append(os.path.join(image_folder,p ))
                    mask_path = [ os.path.join(image_folder, item["mask_path"])]
                    target_img =os.path.join(image_folder,  item["img_path"][-1])
                res = {
                    "id": id,
                    "image_path": [*ref_images,target_img],
                    "mask_path":mask_path,
                    # "image":[image,ref_image],
                    # "gt_mask":gt_mask,
                    "task_name": item["task_name"],
                    "cls_name": item["cls_name"],
                    "dataset_name": item["dataset_name"],
                    # "solution": str(solution),
                }
                if item.get("problem") is not None:
                    res["problem"] = item["problem"]
                all_data.append(res)
                id += 1
    return all_data


def draw_bboxes(image, bboxes):
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        if bbox is not None and len(bbox)==4:
            draw.rectangle(bbox, outline='red', width=2)
    # image.show()
    return draw

def build_prompt(img_num,problem,system_template):
    return [{
        "role": "system",
        "content": [{"type": "text", "text": system_template}],
    }, {
        'role': 'user',
        'content': [
            *({'type': 'image', 'text': None} for _ in range(img_num)),
            {'type': 'text', 'text': problem}
        ]
    }
    ]
def gen_problem_by_dataset(sample):
    dataset_name = sample["dataset_name"]
    cls_name = sample["cls_name"]
    problem_list = {
        "CoSOD3k1024": "Common object segmentation",
        "fewshot1000": "Common object segmentation",
        "LVISRare": "{cls_name}",
        "LVISCommon": "{cls_name}",
        "iNaturalist": "{cls_name}",
        "SASilver_inaturalist": "{cls_name}",
        "Shadow_detection": "shadow",
        "transparent1024": "transparent",
        "rare": "{cls_name}",
        "ultra_rare": "{cls_name}",
        "coco2014_Artifact": "{cls_name}",
        "coco2014_Living": "{cls_name}",
        "cityscape": "{cls_name}",
        "COD10K1024": "camouflaged object segmentation",
        "DUTS": "salient object segmentation",
        "Breast_Tumor": "breast tumor segmentation",
        "isic2018": "skin melanocytic tumor segmentation",
        "Polyp": "polyp colon segmentation",
        "ESDIDefects": "industrial defect segmentation",
    }
    if "MIG" in dataset_name or dataset_name=="MGrounding":
        return sample["problem"]
    problem = problem_list[dataset_name]
    if "cls_name" in problem:
        cls_name = cls_name.replace(dataset_name, "")
        cls_name = cls_name.replace("_", " ")
        problem = problem.format(cls_name=cls_name)
    return problem
from open_r1.rewards import cal_iou


class ConceptSegDataset(torch.utils.data.Dataset):
    '''
    Each item corresponds to a referring mask.
    During __getitem__, one referring sentence is randomly selected from multiple candidates.
    '''


    def __init__(
            self,
            script_args,
            mode="train",
            dataset_names=None,
    ):
        super().__init__()
        self.script_args = script_args
        print("self.script_args :",self.script_args ,)
        self.mode = mode
        self.system_prompt_template = system_prompt_registry["default"]
        self.question_template = question_template_registry[script_args.question_template]
        self.datas = get_cd_data([script_args.data_file_paths], [script_args.image_folders], mode, dataset_names)
        print("dataset_names：", dataset_names)

    def __len__(self):
        return len(self.datas)
    def gen_ref_mig(self,example,RESIZE_SIZE):
        ref_pairs = []
        size = [s // 2 for s in list(RESIZE_SIZE)]
        for i in range(len(example["image_path"]) - 1):
            ref_images = TF.resize(PILImage.open(example["image_path"][i]).convert("RGB"), size)
            ref_pairs.append(ref_images)

        mosaic_ref_images, _ = self.construct_mosaic(RESIZE_SIZE, None, ref_pairs)
        if example.get("ref_box") is not None:
            boxes=example.get("ref_box")
            # %分制坐标
            mosaic_ref_bboxs =  [[box[0]*size[0],box[1]*size[1],box[2]*size[0],box[3]*size[1]] for box in boxes]
        else:
            mosaic_ref_bboxs = []
        return mosaic_ref_images, mosaic_ref_bboxs
    def gen_ref_nomig(self,example,RESIZE_SIZE):
        ref_pairs = []
        ref_bboxs = []
        size = [s // 2 for s in list(RESIZE_SIZE)]
        for i in range(len(example["image_path"]) - 1):
            ref_images = TF.resize(PILImage.open(example["image_path"][i]).convert("RGB"), size)
            ref_masks = TF.resize(PILImage.open(example["mask_path"][i]).convert("L"), size)
            ref_box = process_mask(np.array(ref_masks), example["mask_path"][0], "visual_out/process_mask")
            ref_bboxs.append(ref_box)
            ref_pairs.append(ref_images)

        mosaic_ref_images, mosaic_ref_bboxs = self.construct_mosaic(RESIZE_SIZE,ref_bboxs,ref_pairs)
        return mosaic_ref_images,mosaic_ref_bboxs
    def construct_mosaic(self,RESIZE_SIZE,ref_bboxs,ref_pairs):
        size = [s // 2 for s in list(RESIZE_SIZE)]
        mosaic_ref_images = PILImage.new('RGB', RESIZE_SIZE)
        mosaic_ref_bboxs = []
        if ref_bboxs is None:
            ref_bboxs = [[0,0,0,0]]*len(ref_pairs)
        for i, (bbox, pair) in enumerate(zip(ref_bboxs, ref_pairs)):
            row = i // 2
            col = i % 2
            H, W = size
            ref_images = ref_pairs[i]
            mosaic_ref_images.paste(ref_images, (col * H, row * W))
            # pair = PILImage.new('RGB',(ref_images.width*2,ref_images.height))
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                bbox = [x1 + W * col, y1 + H * row, x2 + W * col, y2 + H * row]
            mosaic_ref_bboxs.append(bbox)
        return mosaic_ref_images,mosaic_ref_bboxs
    def build_meta_reference(self,example, RESIZE_SIZE):
        question = gen_problem_by_dataset(example)
        question_template = self.question_template
        miss_ref_bbox = []
        if len(example["image_path"])==2:# and example["dataset_name"] in ["MIG","MGrounding"]:
            mosaic_ref_images = TF.resize(PILImage.open(example["image_path"][0]).convert("RGB"), RESIZE_SIZE)
            check_answer = "<rule>Visual rule of the reference targets</rule>"
            problem = question_template.format(ref_bboxes="",question=question,check_prompt="",check_answer=check_answer)
        else:

            if  "MIG" in example["dataset_name"] or example["dataset_name"]=="MGrounding":
                mosaic_ref_images,mosaic_ref_bboxs = self.gen_ref_mig(example,RESIZE_SIZE)
            else:
                mosaic_ref_images,mosaic_ref_bboxs = self.gen_ref_nomig(example,RESIZE_SIZE)
            check_answer = "<rule>Visual rule of the reference targets</rule>"
            if len(mosaic_ref_bboxs)==0:
                ref_bboxs = ""
                check_prompt =""
            else:
                miss_ref_bbox = mosaic_ref_bboxs[0]
                mosaic_ref_bboxs = mosaic_ref_bboxs[1:]
                draw_bboxes(mosaic_ref_images,mosaic_ref_bboxs)
                check_prompt="and check it by locating the missed bounding box in the top-left region"
                check_answer+="<check>[x1, y1, x2, y2]</check>"
                ref_bboxs = f": Bounding boxes at {mosaic_ref_bboxs}"

            problem= question_template.format(ref_bboxes=ref_bboxs,question=question,check_prompt=check_prompt,check_answer=check_answer)
        prompt =   build_prompt(2,problem,self.system_prompt_template)



        return miss_ref_bbox,mosaic_ref_images,  prompt,question
    def __getitem__(self, idx):
        data = self.datas[idx]

        min_pixels = self.script_args.min_pixels
        max_pixels = self.script_args.max_pixels
        size =int(math.sqrt(max_pixels))
        miss_ref_bbox,ref_images, prompt,question = self.build_meta_reference(data,  RESIZE_SIZE=(size,size))

        sam_image = image = PILImage.open(data["image_path"][-1]).convert("RGB")
        resized_width=resized_height =size
        image = image.resize((resized_width, resized_height))





        mask = PILImage.open(data["mask_path"][-1]).convert("L").resize((resized_width, resized_height))
        box = process_mask(np.array(mask), data["mask_path"][-1], "visual_out/process_mask")


        gt_mask_np = np.array(mask) > 127
        gt_mask_np = gt_mask_np.astype(np.uint8)  # convert to np.uint8
        gt_mask = torch.from_numpy(gt_mask_np)
        return {
            "image": [ref_images, image],
            "question":question,
            "solution": box,
            "miss_ref_bbox":miss_ref_bbox,
            "prompt": prompt,
            "size": image.size,
            "task_name": data["task_name"],
            "sam_image": sam_image,
            "mask": gt_mask,
            "mask_path": data["mask_path"][-1],
            "image_path": data["image_path"],
        }

