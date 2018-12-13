import tf_pose
import json
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_name")
    parser.add_argument("--cloth_name")
    return parser.parse_args()


def get_pose(image_dir, img_name, pose_dir):
    inference = tf_pose.infer(image=os.path.join(image_dir, img_name))[0]
    body_parts = inference.body_parts

    person_data = []
    for i in range(18):
        try:
            person = body_parts[i]
            x = person.x * 192
            y = person.y * 256
            confidence = person.score
        except:
            x = 0
            y = 0
            confidence = 0
        data = [x, y, confidence]
        person_data += data
    output = {}
    output['version'] = 1.0
    people = dict()
    people['face_keypoints'] = []
    people['pose_keypoints'] = person_data
    people['hand_right_keypoints'] = []
    people['hand_left_keypoints'] = []
    _people = [people]
    output['people'] = _people
    with open(os.path.join(pose_dir, img_name.replace('.jpg', '_keypoints.json')), 'w') as file:
        json.dump(output, file)


if __name__ == "__main__":
    image_dir = os.path.join(os.getcwd(), 'images')  # todo: set this to the directory with images
    pose_dir = os.path.join(os.getcwd(), 'output')
    images = os.listdir(image_dir)
    for i, image in enumerate(images):
        print(i, len(images))
        get_pose(image_dir, image, pose_dir)