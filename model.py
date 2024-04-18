import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# クラス名
# 日本語
classes_ja = ["ひなぎく", "タンポポ", "バラ", "ひまわり", "チューリップ"]
# 英語
classes_en = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
# クラスの数
n_class = len(classes_ja)
# ユーザーが入力する画像サイズ
# 訓練済みのmodelへの入力サイズに合わせる
img_size = 224

# 画像を予測する関数を定義
# imgはユーザーが入力した画像
def predict(img):
    # 画像をrgbに変換
    img = img.convert("RGB")
    # 画像を224x224に変換
    # 訓練したモデルと同じ画像サイズ
    # img = img.resize((img_size, img_size))
    # 訓練したモデルの評価画像と同じ処理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 画像に上記の処理を行う
    img = transform(img)

    # 画像をモデルに入力可能な形に変換
    # batch_sizex, num_ch, w, h
    input = img.reshape(1, 3, img_size, img_size)

    # 訓練済みモデルの呼び出し
    net = torch.load("./model_dense.pth", map_location=torch.device("cpu"))

    # 予測
    net.eval()
    pred = net(input)

    # 結果を確率で返す
    # squeezeでバッチの次元を取り除いている
    pred_prob = torch.nn.functional.softmax(torch.squeeze(pred), dim=0)
    # 降順に並び替える
    sorted_prob, sorted_idx = torch.sort(pred_prob, descending=True)
    return [(classes_ja[idx], classes_en[idx], prob.item()) for idx, prob in zip(sorted_idx, sorted_prob)]
