from mmpretrain import ImageClassificationInferencer

inferencer = ImageClassificationInferencer('./projects/fruits/resnet18_fintuneM.py',
                                           pretrained='./projects/fruits/exp/epoch_10.pth')

image_list = ['data/apple.jpeg', 'data/banana.jpeg', 'data/fruit.jpeg', 'data/grapes.jpg']

for i in range(len(image_list)):
    # result0 = inferencer(image_list[i], show=True)
    result0 = inferencer(image_list[i])
    print(f"file name: {image_list[i]}")
    print(result0[0][list(result0[0].keys())[1]])
    print(result0[0][list(result0[0].keys())[3]])
    print()

results = inferencer(image_list, batch_size=4)

print_keys = list(results[0].keys())
for i in range(len(image_list)):
    # result0 = inferencer(image_list[i], show=True)
    print(f"file name: {image_list[i]}")

    print(results[i][print_keys[1]])
    print(results[i][print_keys[3]])
    print()

