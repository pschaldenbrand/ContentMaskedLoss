import argparse
from paint import *
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='Generate instruction files needed for Aurduino robot painter.')

parser.add_argument('--img', type=str, help='Path of the image to paint.')
parser.add_argument('--actor', type=str, help='Path to the actor.pkl file to use to generate strokes.')
parser.add_argument('--renderer', type=str, help='Filename of renderer used to paint')

# parser.add_argument('--resume', action='store_true', help='Resume training from file name')
parser.add_argument('--divide', default=2, type=int, help='How many times to divide canvas when painting.')
parser.add_argument('--max_step', default=40, type=int, help='Number of steps strange calculation necessary for strokes')
    
args = parser.parse_args()


actions_whole, actions_divided, all_canvases, final_result \
        = paint(args.actor, args.renderer, 
          max_step=3, img=args.img, div=2, discrete_colors=True)

save_actions(actions_whole, actions_divided)

fig, ax = plt.subplots()
ax.imshow(final_result)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(os.path.join(actions_dir, 'painting.png'))