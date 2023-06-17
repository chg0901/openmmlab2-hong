
from mmengine import Config
import matplotlib.pyplot as plt
import mmcv
from mmengine.runner import Runner
from mmseg.utils import register_all_modules
from mmseg.apis import init_model, inference_model, show_result_pyplot

cfg = Config.fromfile('pspnet-Watermelon.py')
register_all_modules(init_default_scope=False)
runner = Runner.from_cfg(cfg)

checkpoint_path = '/home/zeyuan.yin/OpenMMLabCamp/homework-4/log/Watermelon/iter_3000.pth'
model = init_model(cfg, checkpoint_path, 'cuda:0')

img_path = '../test_img/watermelon.png'
img = mmcv.imread(img_path)
result = inference_model(model, img)
pred_mask = result.pred_sem_seg.data[0].cpu().numpy()

plt.imshow(pred_mask)
save_path = img_path.replace('.png', '_pred.png')
plt.savefig(save_path, bbox_inches = 'tight')
print('save_path: ', save_path)