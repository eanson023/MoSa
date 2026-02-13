cd /data
rm -rf checkpoints
mkdir checkpoints
cd checkpoints
echo -e "Downloading pretrained models for Motion-X and HumanML3D datasets"
gdown --fuzzy https://drive.google.com/file/d/1jX0hRKedBXJuwEQQaec7dU1VOFbPpO5U/view?usp=sharing
unzip checkpoints.zip
rm checkpoints.zip
cd /home/user/app