import torch
from torchvision import transforms
from BaseCNN_all import BaseCNN_kmeans, BaseCNN_bn, SCNN
from utils import parse_config
from Transformers import AdaptiveResize
from PIL import Image
import os
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    AdaptiveResize(768),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

config = parse_config()
config.backbone = 'resnet18'
config.representation = 'gap'
config.train = False
base_ckpt_path = config.ckpt_path
#config.base_ckpt_path = os.path.join(base_ckpt_path, str(config.split))

#config.base_ckpt_path = os.path.join(base_ckpt_path, 'bn')


#some global setting
dataset = ['live', 'csiq', 'bid', 'clive', 'koniq10k', 'kadid10k']
default_id2dataset = {}
default_dataset2id = {}
task_id = [0, 1, 2, 3, 4, 5]
for id, task in zip(task_id, dataset):
    default_id2dataset[id] = task
    default_dataset2id[task] = id

task_id = [0, 1, 2, 3, 4, 5]
seq_len = len(dataset)
eval_dict = {"live": False,
             "csiq": False,
             "bid": False,
             "clive": False,
             "koniq10k": False,
             "kadid10k": False}

trainset_dict = {"live": config.live_set,
                 "csiq": config.csiq_set,
                 "bid": config.bid_set,
                 "clive": config.clive_set,
                 "koniq10k": config.koniq10k_set,
                 "kadid10k": config.kadid10k_set}

cluster_dict = {"live": 128,
                 "csiq": 128,
                 "bid": 128,
                 "clive": 128,
                 "koniq10k": 128,
                 "kadid10k": 128}
config.reverse = False
if config.reverse:
    dataset.reverse()

id2dataset = {}
dataset2id = {}
current2default = {}
for id, task in zip(task_id, dataset):
    id2dataset[id] = task
    dataset2id[task] = id
    current2default[id] = default_dataset2id[task]
config.id2dataset = id2dataset
config.dataset2id = dataset2id

config.current2default = current2default

#base_ckpt_path = os.path.join(config.ckpt_path,'bn_exp')

base_ckpt_path = os.path.join(config.ckpt_path,'1')

config.base_ckpt_path = base_ckpt_path

model = BaseCNN_bn(config).cuda()

ckpt = './checkpoint/bn_exp/1/train_on_kadid10k/BaseCNN_bn_best.pt'

checkpoint = torch.load(ckpt)
model.load_state_dict(checkpoint['state_dict'])

# scnn = SCNN()
# scnn = torch.nn.DataParallel(scnn)
# scnn.to(device)
# scnn_root = os.path.join('saved_weights', 'scnn.pkl')
# scnn.load_state_dict(torch.load(scnn_root))
# scnn = scnn
# scnn.eval()

#image1 = '/home/redpanda/codebase/IQA_Database/BID/ImageDatabase/DatabaseImage0204.JPG'
#image1 = '/home/redpanda/codebase/IQA_Database/CSIQ/dst_imgs/jpeg2000/sunset_sparrow.jpeg2000.4.png'
#image1 = '/home/redpanda/codebase/IQA_Database/ChallengeDB_release/Images/121.bmp'
image1 = '/home/redpanda/codebase/IQA_Database/kadid10k/images/I12_19_05.png'
#image1 = '/home/redpanda/codebase/IQA_Database/databaserelease2/fastfading/img37.bmp'

#image1 = '/home/redpanda/codebase/IQA_Database/koniq-10k/1024x768/6645063001.jpg'

image1 = Image.open(image1)
image1 = test_transform(image1)
image1 = torch.unsqueeze(image1, dim=0)
image1 = image1.to(device)


weights = []
weights1 = []
weights2 = []
weights3 = []
weights4 = []

y_pred = torch.zeros(6)
for i in range(6):
    task_folder = 'train_on_' + config.id2dataset[i]
    config.task_id = i

    bn_path = os.path.join(config.base_ckpt_path, task_folder, 'distortion_bn.pt')
    model.load_bn(bn_path)
    model.eval()
    with torch.no_grad():
        _, feat = model(image1)
    feat1 = feat[0]
    feat2 = feat[1]
    feat3 = feat[2]
    feat4 = feat[3]
    #sfeat = scnn(image1)

    bn_path = os.path.join(config.base_ckpt_path, task_folder, 'saved_bn.pt')
    model.load_bn(bn_path)
    model.eval()
    with torch.no_grad():
        y_bar, _ = model(image1)

    cluster_path = os.path.join(config.base_ckpt_path, task_folder, 'cluster_sk1.pt')
    centroids = torch.load(cluster_path)
    D = euclidean_distances(feat1.cpu().numpy(), centroids.numpy())
    D = np.min(D)
    D = torch.from_numpy(np.array(D)).cuda()
    weights1.append(D)

    cluster_path = os.path.join(config.base_ckpt_path, task_folder, 'cluster_sk2.pt')
    centroids = torch.load(cluster_path)
    D = euclidean_distances(feat2.cpu().numpy(), centroids.numpy())
    D = np.min(D)
    D = torch.from_numpy(np.array(D)).cuda()
    weights2.append(D)

    cluster_path = os.path.join(config.base_ckpt_path, task_folder, 'cluster_sk3.pt')
    centroids = torch.load(cluster_path)
    D = euclidean_distances(feat3.cpu().numpy(), centroids.numpy())
    D = np.min(D)
    D = torch.from_numpy(np.array(D)).cuda()
    weights3.append(D)

    cluster_path = os.path.join(config.base_ckpt_path, task_folder, 'cluster_sk4.pt')
    centroids = torch.load(cluster_path)
    D = euclidean_distances(feat4.cpu().numpy(), centroids.numpy())
    D = np.min(D)
    D = torch.from_numpy(np.array(D)).cuda()
    weights4.append(D)

    hidx = config.current2default[i]
    y_pred[i] = y_bar[hidx][0][0]

weights1 = torch.stack(weights1)
weights1 = F.softmin(weights1 * 32, dim=0)

weights2 = torch.stack(weights2)
weights2 = F.softmin(weights2 * 32, dim=0)

weights3 = torch.stack(weights3)
weights3 = F.softmin(weights3 * 32, dim=0)

weights4 = torch.stack(weights4)
weights4 = F.softmin(weights4 * 32, dim=0)

weights = (weights3 + weights4) / 2

y_bar = torch.dot(weights, y_pred.cuda())

print(weights)
print(y_pred)
print(y_bar)


# #print(weights1)
# #print(weights2)
# print(weights6.data)
# # print(weights4)
# # print(weights5)
# print(torch.dot(weights, y_pred.cuda()))
# print(torch.dot(weights6, y_pred.cuda()))


