# 動画ファイルを入力し、物体検出を行い、
# 開始時刻[秒](start_sec)からフレーム数(frame_num)分だけ
# 動画ファイル(x264)へ出力する。

import os
import sys
import cv2
import torch

in_file   = "2022_0121_090959_863.MOV"	#入力ファイル名
out_file  = "output.mp4"		#出力ファイル名
start_sec = 154				#開始時刻[秒]
frame_num = 60				#出力するフレーム数


#モデルの読み込みと設定
model = torch.hub.load('ultralytics/yolov5','yolov5s')
#model = torch.hub.load('ultralytics/yolov5','yolov5m')

model.conf = 0.3	#検出の下限値
#model.classes = [0]	#0:person クラスだけ
#print(model.names)


#動画ファイルの読み込み
camera = cv2.VideoCapture(in_file)


#キャプチャ開始位置の設定と幅・高さの取得
fps = camera.get(cv2.CAP_PROP_FPS)
w   = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
camera.set(cv2.CAP_PROP_POS_FRAMES, round(fps * start_sec))


#出力ファイルのフォーマット・大きさ指定
fourcc = cv2.VideoWriter_fourcc(*'X264')
out    = cv2.VideoWriter(out_file, fourcc, fps, (w,h))


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

  #動画の出力
  out.write(imgs)

  #表示
  cv2.imshow('color',imgs)

  frame += 1
  if frame > frame_num:
    break

  #"q"を押すと終了
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break


camera.release()
out.release()
cv2.destroyAllWindows()
