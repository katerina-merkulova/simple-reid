# https://github.com/Kaggle/kaggle-api
pip install --user kaggle
mkdir -p ~/.kaggle
echo '{"username":"ekaterinamerkulova","key":"fe9be514feb8a9aef83c08f5836b85f1"}' > ~/.kaggle/kaggle.json
sudo chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download pengcw1/market-1501 -p data/Market --unzip

# python -m pip install -r requirements.txt
# python main.py --dataset market1501 --cfg configs/res50_ce_triplet.yaml --gpu 0,1