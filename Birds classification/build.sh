set -o xtrace

setup_root() {
    apt-get install -qq -y \
        python3-pip \
        python3-tk \
        git

    pip3 install -qq \
        pytest \
        scikit-image \
        scikit-learn \
        opencv-python \
        matplotlib \
        imgaug \
        pandas \
        pytorch-ignite \
        albumentations==1.0.3 \
        torch==1.9.1 \
        torchvision==0.10.1 \
        pytorch_lightning==1.2.0 \
        efficientnet_pytorch==0.7.1 \
        albumentations_experimental

    pip3 install -qq git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
}

setup_checker() {
    python3 -c 'import matplotlib.pyplot'
}

"$@"