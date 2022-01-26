# 動画ファイルを入力し、物体検出を行い、
# 開始時刻[秒](start_sec)からフレーム数(frame_num)分だけ
# フレーム毎にJPEGファイルへ出力する。

import os
import sys
import cv2
import torch

out_dir   = "out"			#出力ディレクトリ
in_file   = "2022_0121_090959_863.MOV"	#入力ファイル名
start_sec = 154				#開始時刻[秒]
frame_num = 60				#出力するフレーム数


#出力ディレクトリの作成
os.makedirs(out_dir, exist_ok=True)


#モデルの読み込みと設定
model = torch.hub.load('ultralytics/yolov5','yolov5s')
#model = torch.hub.load('ultralytics/yolov5','yolov5m')

model.conf = 0.3	#検出の下限値
#model.classes = [0]	#0:person クラスだけ
#print(model.names)


#動画ファイルの読み込み
camera = cv2.VideoCapture(in_file)


#キャプチャ開始位置の設定
fps = camera.get(cv2.CAP_PROP_FPS)
camera.set(cv2.CAP_PROP_POS_FRAMES, round(fps * start_sec))


#動画すべてのフレームで処理をする
frame = 0
while True:

  ret, imgs = camera.read()
  if not ret :
    while cv2.waitKey(100) == -1:	#動画ファイルの最後ではキーを押すと終了
      pass
    break

  results = model(imgs)		#default=640

  #検出情報の描画
  for *bb, conf, cls in results.xyxy[0]:

      s = model.names[int(cls)]+":"+'{:.1f}'.format(float(conf)*100)

      cc = (255,255,0)
      cc2 = (128,0,0)

      cv2.rectangle(
          imgs,
          (int(bb[0]), int(bb[1])),
          (int(bb[2]), int(bb[3])),
          color=cc,
          thickness=2,
          )

      cv2.rectangle(imgs, (int(bb[0]), int(bb[1])-20), (int(bb[0])+len(s)*10, int(bb[1])), cc, -1)
      cv2.putText(imgs, s, (int(bb[0]), int(bb[1])-5), cv2.FONT_HERSHEY_PLAIN, 1, cc2, 1, cv2.LINE_AA)

  #JPEGファイルの出力
  cv2.imwrite('{}/{}.jpg'.format(out_dir,str(frame).zfill(4)), imgs)

  #表示
  cv2.imshow('color',imgs)

  frame += 1
  if frame > frame_num:
    break

  #"q"を押すと終了
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break


camera.release()
cv2.destroyAllWindows()
