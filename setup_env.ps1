conda create -n RGCL python=3.10 -y
conda install -n RGCL pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install -n RGCL -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl -y
conda run -n RGCL pip install -r requirement.txt
