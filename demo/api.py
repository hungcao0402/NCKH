import numpy as np
from fastapi import FastAPI, HTTPException

app = FastAPI()

from hhcl import models
from hhcl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from torch import nn
from hhcl.utils import to_torch
from hhcl.utils.data import transforms as T
from PIL import Image
import torch

def extract_cnn_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    outputs = model(inputs)
    outputs = outputs.data.cpu()
    return outputs


def load_image(img_path, height=256, width=128):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    img = Image.open(img_path).convert('RGB')
    img = test_transformer(img)
    return img

def create_model(checkpoint_path: str='./model_85.pth.tar'):
    # Create model
    model = models.create('resnet50', pretrained=False, num_features=0, dropout=0,
                          num_classes=0, pooling_type='gem')

    # Load from checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    copy_state_dict(checkpoint['state_dict'], model, strip='module.')
    
    model.cuda()
    model = nn.DataParallel(model)

    # Evaluator
    model.eval()
    return model

market_model = create_model()

dataset_dir = 'database'
def load_data(dataset: str='market1501'):
    '''
    Load data from npy files
    return: query, gallery, distmat
    '''
    query = np.load(f'{dataset_dir}/{dataset}/query.npy')
    gallery = np.load(f'{dataset_dir}/{dataset}/gallery.npy')
    distmat = np.load(f'{dataset_dir}/{dataset}/distmat.npy')

    # query_features= np.load('query_features.npy')
    gallery_features= np.load(f'{dataset_dir}/{dataset}/gallery_features.npy')

    # query_paths, query_ids,  query_cams = np.hsplit(query, 3)
    # gallery_paths, gallery_ids, gallery_cams = np.hsplit(gallery, 3)
    # # gallery_paths = [i.split('/')[-3] for i in gallery_paths]
    # gallery_paths = np.squeeze(gallery_paths)
    # gallery_ids = np.squeeze(gallery_ids)

    index = np.argsort(distmat, axis=1) # from small to large
    print(f'Done load {dataset}!!!!')

    return index, gallery, distmat, query, gallery_features

market_data = load_data('market1501')
# msmt_data = load_data('msmt17')
market_gallery_features = market_data[-1]
# msmt_gallery_features = msmt_data[-1]


def search( file_path, index, gallery, distmat, topk=20):
    outputs = []
    img = load_image(file_path)
    feature = extract_cnn_feature(market_model, img.unsqueeze(0))
    distmat = feature @ market_gallery_features.T
    index = np.argsort(distmat, axis=1)
    index = torch.flip(index, [1])[0]
    for i in index:
        img_path = gallery[i][0]
        g_pid = gallery[i][1]
        outputs.append({'img_path': img_path,
                       'g_pid': int(g_pid),
                       'cam_id': int(gallery[i][2])})
        if len(outputs) >= topk:
            break
    return outputs


@app.get("/search")
def retrieval(file: str='database/test.jpg',dataset: str = 'market1501',  topk: int = 20):
    if dataset.lower() == 'market1501':
        outputs = search( file,market_data[0], market_data[1], market_data[2], topk)
        return outputs

    elif dataset.lower() == 'msmt17':
        outputs = search( file,market_data[0], market_data[1], market_data[2], topk)
        return outputs
    else:
        raise HTTPException(status_code=400, detail="Dataset not supported")

