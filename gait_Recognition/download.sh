wget "http://www.cbsr.ia.ac.cn/GaitDatasetB-silh.zip"

for f in *.tar.gz; do 
tar -xvf "$f"; 
done 

# Download C3D pretrained Model- Pretrained on Sports 1-M dataset
wget http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle