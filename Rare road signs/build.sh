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
        albumentations==1.1.0 \
        torch==1.10.0 \
        torchvision==0.11.1 \
        pytorch_lightning==1.5.0 \
        efficientnet_pytorch \
        albumentations_experimental \
        timm==0.4.12

    pip3 install -qq git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
}

setup_checker() {
    python3 -c 'import matplotlib.pyplot'
}

"$@"