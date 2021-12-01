
import argparse
import collections
import json
import os


def preprocess_tag(tag: str):
    return tag if tag.split(' ') == 1 else '_'.join(tag.split(' '))

def create_image_to_tags(instances, karpathy):
    id2tag   = {cat['id']: preprocess_tag(cat['name']) for cat in instances['categories']} 
    karp_ids = set([row['cocoid'] for row in karpathy['images']])
    img2tags = collections.defaultdict(list)
    for row in instances['annotations']:
        img = row['image_id']
        if img in karp_ids:
            tag = id2tag[row['category_id']]
            img2tags[img].append(tag)
    return img2tags



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--coco_annotations_dir',
        type=str,
        help='Path to karpathy annotations file'        
    )
    args = parser.parse_args()

    coco_train = os.path.join(args.coco_annotations_dir, 'instances_train2014.json')
    coco_val   = os.path.join(args.coco_annotations_dir, 'instances_val2014.json')
    karpathy   = os.path.join(args.coco_annotations_dir, 'karpathy_coco_split.json')
    with open(coco_train, 'r') as fp:
        instances = json.load(fp)
    with open(coco_val, 'r') as fp:
        instances['annotations'] += json.load(fp)['annotations']
    with open(karpathy, 'r') as fp:
        karpathy = json.load(fp)

    img2tags = create_image_to_tags(instances, karpathy)
    
    out_file = os.path.join('cache', 'image_to_tags.json')
    with open(out_file, 'w') as fp:
        json.dump(img2tags, fp)
    print('Image to tags mapping successfully saved in {}'.format(out_file))
