set -o xtrace

setup_root() {
    apt-get install -qq -y \
        python3-pip \
        python3-tk

    pip3 install -qq \
        pytest \
        scikit-image \
        scikit-learn \
        matplotlib \
        albumentations==1.0.3 \
        torch==1.9.1 \
        torchvision==0.10.1 \
        pytorch_lightning==1.2.0
}

setup_checker() {
    python3 -c 'import matplotlib.pyplot'
}

"$@"