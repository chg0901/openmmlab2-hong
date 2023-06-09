from mmpretrain import ImageClassificationInferencer
import cv2
import matplotlib.pyplot as plt
inferencer = ImageClassificationInferencer('./resnet50_fintuneM2.py',
                                           pretrained='./exp3_resnet50/best_accuracy_top1_epoch_8.pth')

image_list = ['data/apple.jpeg', 'data/banana.jpeg', 'data/fruit.jpeg', 'data/grapes.jpg']
image_list = ['../../'+i for i in image_list]

fig, _ = plt.subplots(nrows=2, ncols=2,figsize=(15, 15))
for i, ax in enumerate(fig.axes):
    img = cv2.imread(image_list[i])[:, :, (2, 1, 0)]
    # img=cv2.imread('outputs_test_data/apple.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.axis("off")
    ax.imshow(img)

plt.savefig("all_test_image.png", bbox_inches='tight')
plt.show()


# # infer each image
# for i in range(len(image_list)):
#     # result0 = inferencer(image_list[i], show=True)
#     result0 = inferencer(image_list[i])
#     print(f"file name: {image_list[i]}")
#     print(result0[0][list(result0[0].keys())[1]])
#     print(result0[0][list(result0[0].keys())[3]])
#     print()

# infer all image in the image_list
results = inferencer(image_list, batch_size=4)

print_keys = list(results[0].keys())

fig, _ = plt.subplots(nrows=2, ncols=2,figsize=(15, 15))
for i, ax in enumerate(fig.axes):
    image_name = image_list[i].split('/')[-1].split('.')[0]
    img = cv2.imread(f'outputs_test_data/{image_name}.png')[:, :, (2, 1, 0)]
    # img=cv2.imread('outputs_test_data/apple.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.axis("off")
    ax.imshow(img)
    print(f"file name: {image_list[i]}")
    print(f"{print_keys[1]}: {results[i][print_keys[1]]}")
    print(f"{print_keys[3]}: {results[i][print_keys[3]]}")
    print()

plt.savefig("all_predicted_image.png", bbox_inches='tight')
plt.show()

# # python  ../../demo/image_demo.py  ../../data/grapes.jpg  resnet50_fintuneM2.py  --checkpoint exp3_resnet50/best_accuracy_top1_epoch_8.pth  --show-dir outputs_test_data/
# # python  \
# #     ../../demo/image_demo.py  \
# #     ../../data/grapes.jpg  \
# #     resnet50_fintuneM2.py  \
# #     --checkpoint exp3_resnet50/best_accuracy_top1_epoch_8.pth \
# #     --show-dir outputs_test_data/

# (openmmlab-pose) cine@cine-prof:~/Documents/GitHub/mmpretrain/projects/fruits$  python  ../../demo/image_demo.py  ../../data/grapes.jpg  resnet50_fintuneM2.py  --checkpoint exp3_resnet50/best_accuracy_top1_epoch_8.pth  --show-dir outputs_test_data/
# Loads checkpoint by local backend from path: exp3_resnet50/best_accuracy_top1_epoch_8.pth
# Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
# 06/09 12:43:07 - mmengine - WARNING - `Visualizer` backend is not initialized because save_dir is None.
# {
#   "pred_label": 24,
#   "pred_score": 0.9906376004219055,
#   "pred_class": "葡萄-红"
# }

# # 查看预测出来的图片
# # import cv2
# # import matplotlib.pyplot as plt
# # image_list = ['data/apple.jpeg', 'data/banana.jpeg', 'data/fruit.jpeg', 'data/grapes.jpg']
# img=cv2.imread('outputs_test_data/apple.png')[:,:,(2,1,0)]
# # img=cv2.imread('outputs_test_data/apple.png')
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# # img=cv2.imread('outputs_test_data/banana.png')
# # img=cv2.imread('outputs_test_data/fruit.png')
# # img=cv2.imread('outputs_test_data/grapes.png')
# plt.imshow(img)
# plt.show()


# python  ../../demo/image_demo.py  ../../data/apple.*  resnet50_fintuneM2.py  --checkpoint exp3_resnet50/best_accuracy_top1_epoch_8.pth  --show-dir outputs_test_data/
# python  ../../demo/image_demo.py  ../../data/ban*.*  resnet50_fintuneM2.py  --checkpoint exp3_resnet50/best_accuracy_top1_epoch_8.pth  --show-dir outputs_test_data/
# python  ../../demo/image_demo.py  ../../data/fru*.*  resnet50_fintuneM2.py  --checkpoint exp3_resnet50/best_accuracy_top1_epoch_8.pth  --show-dir outputs_test_data/
# python  ../../demo/image_demo.py  ../../data/grapes.jpg  resnet50_fintuneM2.py  --checkpoint exp3_resnet50/best_accuracy_top1_epoch_8.pth  --show-dir outputs_test_data/


# Loads checkpoint by local backend from path: exp3_resnet50/best_accuracy_top1_epoch_8.pth
# Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
# 06/09 12:56:30 - mmengine - WARNING - `Visualizer` backend is not initialized because save_dir is None.
# {
#   "pred_label": 18,
#   "pred_score": 0.6414839029312134,
#   "pred_class": "苹果-红"
# }
#
# Loads checkpoint by local backend from path: exp3_resnet50/best_accuracy_top1_epoch_8.pth
# Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
# 06/09 12:56:33 - mmengine - WARNING - `Visualizer` backend is not initialized because save_dir is None.
# {
#   "pred_label": 28,
#   "pred_score": 1.0,
#   "pred_class": "香蕉"
# }
#
# Loads checkpoint by local backend from path: exp3_resnet50/best_accuracy_top1_epoch_8.pth
# Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
# 06/09 12:56:36 - mmengine - WARNING - `Visualizer` backend is not initialized because save_dir is None.
# {
#   "pred_label": 5,
#   "pred_score": 0.3799188733100891,
#   "pred_class": "柠檬"
# }
#
# Loads checkpoint by local backend from path: exp3_resnet50/best_accuracy_top1_epoch_8.pth
# Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
# 06/09 12:56:38 - mmengine - WARNING - `Visualizer` backend is not initialized because save_dir is None.
# {
#   "pred_label": 24,
#   "pred_score": 0.9906376004219055,
#   "pred_class": "葡萄-红"
# }
