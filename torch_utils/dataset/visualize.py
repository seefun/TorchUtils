import cv2
import matplotlib.pyplot as plt

# visualize tools
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
def test_transform(img_path, transform):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(image = img)['image']
    visualize(image = img)
    
def write_aug(img_path, transform, num=30):
    img = cv2.imread(img_path)
    for i in range(num):
        t = transform(image = img)['image']
        cv2.imwrite('./aug/'+str(i)+'.jpg',t)
