
python train.py --debug --batch_size=8 --max_step=120 --renderer=renderer.pkl --resume=pretrained_models/gan --loss_fcn=gan


tensorboard --logdir=train_log --port=6006

python generate_actions.py --img=image/hambone.jpg --max_step=4 --actor=pretrained_models/cml1/actor.pkl --renderer=renderer_constrained.pkl