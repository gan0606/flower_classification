import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import os
# model.pyのpredict関数を使用
from model import predict

# pyplotを使用する際に注記が出ないようにする文
st.set_option("deprecation.showPyplotGlobalUse", False)

# 関数化する
def main():
    # タイトル
    st.title("転移学習による花の画像の分類")
    st.write("最終更新日: 2024/4/18")

    # サイドバーのmenu
    menu = ["概要", "分類結果", "画像シュミレーター"]
    # サイドバーの作成
    chosen_menu = st.sidebar.selectbox(
        "menu選択", menu
    )

    # ファイルの設定
    # 訓練済みのモデルファイル

    # 分類対象の画像
    object_file = "./flowers.jpg"
    # テストデータの結果
    result_file = "./predict_flowers.jpg"


    # 読み込めているかを確認
    is_object_file = os.path.isfile(object_file)
    is_result_file = os.path.isfile(result_file)

    # printで出力すると、ターミナルに出る
    # st.writeだとブラウザ上に出る
    print(is_object_file)
    print(is_result_file)
    
    

    # menuの中身
    # 分析の概要
    if chosen_menu == "概要":
        st.subheader("分析概要")
        st.write("転移学習と呼ばれる手法を活用することで、花の画像を5種類に自動で分類するシステムを作りました。")
        st.write("転移学習について")
        st.write(
            """
            転移学習とは、あるタスクで学習したモデルを別のタスクに適用するディープラーニングの手法です。
            この分析では、一般公開されているDenseNet121というモデルを使用しました。
            DenseNet121は、ImageNetなどの大規模なデータセットで事前学習されたモデルが公開されており、転移学習に適しています。
        """
        )
        st.subheader("データセットの内容")
        st.write("このデータセットは、5種類の異なる花卉（ひなぎく・タンポポ・バラ・ひまわり・チューリップ）の画像が合わせて4,317枚含まれています。")
        st.write("データセットのうち8割をモデルを再訓練するための訓練データとして使用して、2割を評価を行うためのテストデータとして使用しました。")
        st.write(" ")
        st.write(" ")
        st.text("訓練データの一部")
        # 画像の表示
        image_object = Image.open(object_file)
        st.image(image_object)
       
    # 分類の結果
    elif chosen_menu == "分類結果":
        st.subheader("分類結果")
        # 結果の表示
        image_result = Image.open(result_file)
        st.image(image_result)

        # 結果についての説明
        st.write("構築したニューラルネットワークは未知のデータに対して96.4%の精度で分類を行うことができました。")
        st.write("上記の画像はテストデータのうち16枚を可視化したものです。")
        st.write("画像のtrueは正解ラベルを示し、predictはモデルを使用して予測したラベルを示しています。")

    elif chosen_menu == "画像シュミレーター":
        st.subheader("画像シュミレーター")
        st.write("訓練したAIでアップロードされた花の画像を分類します。")
        st.write("分類できる花の種類は、ひなぎく・タンポポ・バラ・ひまわり・チューリップです。")
        # 空白行
        st.write("")
        # ラジオボタンの作成
        img_source = st.radio("画像のソースを選択してください", ("画像をアップロード", "カメラで撮影"))

        # 画像のアップロード
        if img_source == "画像をアップロード":
            # ファイルをアップロード
            img_file = st.file_uploader("画像を選択してください", type=["png", "jpg"])
        # カメラで撮影する場合
        elif img_source == "カメラで撮影":
            # カメラ撮影
            img_file = st.camera_input("カメラで撮影")

        # 推定の処理
        # img_fileが存在する場合に処理を進める
        if img_file is not None:
            # 特定の処理が行われていることを知らせる
            with st.spinner("推定中です..."):
                # 画像ファイルを開く
                img = Image.open(img_file)
                # 画面に画像を表示
                st.image(img, caption="予測対象画像", width=480)

                # 空白行
                st.write("")

                # 予測
                results = predict(img)

                # 結果の表示
                st.subheader("判定結果")
                # 予測確率が高い上位3位
                n_top = 3
                for result in results[:n_top]:
                    st.write(f"{round(result[2]*100, 2)}%の確率で{result[0]}です。")

                # 円グラフの表示
                # 円グラフのラベル
                pie_labels = [result[1] for result in results[:n_top]]
                # 上位3位以外はothorsにする
                pie_labels.append("others")
                # 予測した確率
                pie_probs = [result[2] for result in results[:n_top]]
                # 上位3以外は上位3以外の確率の和を取る
                pie_probs.append(sum(result[2] for result in results[n_top:]))
                # グラフの表示
                fig, ax = plt.subplots()
                # グラフをドーナツ型にする設定
                wedgeprops = {"width":0.3, "edgecolor":"white"}
                # フォントサイズの設定
                text_props = {"fontsize":6}
                ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90, textprops=text_props, autopct="%.2f", wedgeprops=wedgeprops)
                st.pyplot(fig)

# streamlitを実行したときにmain()を実行するという表記
if __name__ == "__main__":
    main()
