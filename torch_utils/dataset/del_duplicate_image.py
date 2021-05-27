import os
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import imagehash

funcs = [
        imagehash.average_hash,
        imagehash.phash,
        imagehash.dhash,
        imagehash.whash,
    ]

def delete_duplicate_imghash(imgpath_list, threshold=0.9, verbose=True):
    image_ids = []
    hashes = []

    for path in tqdm(imgpath_list):
        image = Image.open(path)
        #image_id = os.path.basename(path)
        image_id = path
        image_ids.append(image_id)
        hashes.append(np.array([f(image).hash for f in funcs]).reshape(256))
    
    hashes_all = np.array(hashes)

    sims = np.array([np.sum((hashes_all[i] == hashes_all), axis=1)/256 for i in range(hashes_all.shape[0])])

    indices1 = np.where(sims > threshold)
    indices2 = np.where(indices1[0] != indices1[1])
    image_ids1 = [image_ids[i] for i in indices1[0][indices2]]
    image_ids2 = [image_ids[i] for i in indices1[1][indices2]]
    dups = {tuple(sorted([image_id1,image_id2])):True for image_id1, image_id2 in zip(image_ids1, image_ids2)}
    print('found %d pairs of duplicates' % len(dups))

    duplicate_image_ids = sorted(list(dups))
    if verbose:
        for pair in duplicate_image_ids:
            print('found duplicate image pair:', pair)

    del_list = []
    for pair in duplicate_image_ids:
        del1 = pair[0] not in del_list
        del2 = pair[1] not in del_list
        if del1 and del2:
            del_list.append(pair[1])
            if verbose:
                print(pair[1], 'deleted')
    print('%d of duplicated images deleted' % len(del_list))

    for del_img in del_list:
        imgpath_list.remove(del_img)

    return imgpath_list
