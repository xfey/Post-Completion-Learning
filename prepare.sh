sudo apt-get install git-lfs -y
curl -LsSf https://astral.sh/uv/install.sh | sh
export UV_HTTP_TIMEOUT=6000
make install
. openr1/bin/activate

# pip install httpx==0.23.0
pip install -U byted-wandb -i https://bytedpypi.byted.org/simple   # install wandb by bytedpypi

### huggingface
huggingface-cli login --token "$(cat /mnt/bn/amlocr-fx-storage/workflow/hf_token.txt | tr -d '\n\r ')"

### wandb
# pip install -U byted-wandb -i https://bytedpypi.byted.org/simple
# cp /mnt/bn/amlocr-fx-storage/workflow/.netrc /home/tiger/.netrc
# wandb login
# wandb offline