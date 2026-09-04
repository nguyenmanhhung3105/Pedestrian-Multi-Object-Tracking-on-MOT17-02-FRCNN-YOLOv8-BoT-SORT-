# Pedestrian Multi-Object Tracking on MOT17-02-FRCNN (YOLOv8 + BoT-SORT)

Pipeline theo dõi người đi bộ (pedestrian multi-object tracking) trên sequence MOT17-02-FRCNN, kết hợp detector YOLOv8 với tracker BoT-SORT để duy trì ID xuyên suốt video. Vì detector tạo ra một số false-positive tĩnh lặp lại ở một vùng cố định trong khung hình, pipeline bổ sung một hệ thống lọc false-positive 2 giai đoạn (heuristic tự động + review thủ công) trước khi đánh giá bằng bộ công cụ TrackEval chính thức.

## Kiến trúc pipeline

```
Frame ──► YOLOv8 Detector ──► BoT-SORT Tracker ──► Lọc false-positive (2 giai đoạn) ──► TrackEval (HOTA/MOTA/IDF1)
```

- **Detector**: YOLOv8 (Ultralytics), model weight `yolov8m.pt` khai báo trong code qua biến `MODEL_NAME`. Chỉ giữ lại class `0` (person, theo COCO class index) qua tham số `classes=[0]`
- **Detection threshold**: `conf=0.3`, `iou=0.5`
- **Tracker**: BoT-SORT, chạy qua `model.track(..., tracker="botsort.yaml", persist=True)` — `persist=True` để giữ ID track xuyên suốt các frame liên tiếp

## Dataset

- **Sequence**: MOT17-02-FRCNN (từ MOT17 benchmark), đọc từ `DATASET_DIR/img1/*.jpg` và ground-truth `DATASET_DIR/gt/gt.txt`
- **Số frame**: 600 frame (`.jpg`, sắp xếp theo số thứ tự frame)
- **Độ phân giải**: 1920×1080
- Đường dẫn dataset gốc trong code (môi trường Kaggle): `/kaggle/input/datasets/duyl8787/mot17-02-frcnn/MOT17-02-FRCNN`

## Hệ thống lọc false-positive (2 giai đoạn)

**Giai đoạn 1 — Heuristic tự động theo vùng ("suspect zone"):**
Một vùng đa giác cố định trong khung hình (`SUSPECT_ZONE`, góc trên bên phải khung hình, toạ độ `(1480,250)–(1920,850)`) được xác định là nơi hay phát sinh false-positive tĩnh (VD: quầy hàng/vật thể tĩnh bị nhận nhầm là người). Với mỗi track, tính:
- Tỉ lệ overlap giữa bounding box và vùng nghi vấn (`bbox_overlap_ratio_with_zone`)
- Độ dịch chuyển net (`displacement`, khoảng cách điểm đầu–cuối) và biên độ dao động (`span`, khoảng cách max–min) của tâm đáy bounding box qua các frame

Ngưỡng áp dụng:
- `MIN_STATIC_FRAMES = 8` — track phải xuất hiện tối thiểu 8 frame mới được xét
- `MIN_ZONE_HIT_RATIO = 0.5` — track phải nằm trong vùng nghi vấn ở ≥50% số frame
- `MAX_STATIC_DISPLACEMENT = 45.0` và `MAX_STATIC_SPAN = 60.0` — track bị coi là tĩnh nếu displacement và span đều dưới ngưỡng này

Track thoả cả 2 điều kiện (chủ yếu nằm trong vùng nghi vấn + gần như đứng yên) bị loại toàn bộ.

**Giai đoạn 2 — Review thủ công:**
Danh sách ID đã xác nhận là false-positive qua xem lại video (`CONFIRMED_FALSE_POSITIVE_IDS = {44, 93, 99, 169, 196, 288}`) bị loại bỏ, nhưng **chỉ trong phạm vi vùng nghi vấn** (`overlap_ratio > MIN_ZONE_OVERLAP = 0.20`) để tránh xoá nhầm box hợp lệ nếu track đó có xuất hiện ngoài vùng này.

Một trường hợp đặc biệt được xử lý riêng: **ID 80** ban đầu là false-positive tĩnh trong vùng nghi vấn, sau đó chuyển thành theo dõi người thật. Code tính vị trí "neo" (median toạ độ của `ID80_ANCHOR_FRAMES=10` frame đầu trong vùng), rồi tìm frame mà track di chuyển đủ xa khỏi vị trí neo (`ID80_ACTIVATION_DISTANCE=70.0`, duy trì liên tục `ID80_ACTIVATION_CONSECUTIVE=3` frame) để xác định frame bắt đầu giữ lại track ID 80.

## Cài đặt

```python
[sys.executable, "-m", "pip", "install", "-q",
 "ultralytics", "huggingface_hub", "trackeval-python", "motmetrics", "lap"]
```

Nếu cài `trackeval-python` từ PyPI thất bại, code tự động fallback sang cài TrackEval trực tiếp từ GitHub:

```python
[sys.executable, "-m", "pip", "install", "-q",
 "ultralytics", "huggingface_hub", "motmetrics", "lap",
 "git+https://github.com/JonathonLuiten/TrackEval.git"]
```

Các thư viện khác được import trực tiếp (giả định có sẵn trên môi trường Kaggle, không pin version): `cv2` (opencv-python), `numpy`, `tqdm`.

**Môi trường chạy**: Kaggle Notebook — code phụ thuộc trực tiếp vào `kaggle_secrets.UserSecretsClient` (đọc `HF_TOKEN` từ Kaggle Secrets) và đường dẫn dataset cố định dạng `/kaggle/input/...`, nên không chạy được nguyên trạng ngoài môi trường Kaggle.

## Cách sử dụng

Không có script/CLI riêng — toàn bộ pipeline chạy tuần tự trong 1 file (bản `.py` được export từ notebook gốc). Chạy trực tiếp:

```bash
python 52300102_52300107.py
```

Yêu cầu trước khi chạy:
- Kaggle Secret tên `HF_TOKEN` (token HuggingFace, dùng để cache model YOLO và upload kết quả)
- Dataset Kaggle `duyl8787/mot17-02-frcnn` đã được attach vào notebook/kernel

Pipeline khi chạy sẽ tự động thực hiện tuần tự các bước sau (không có tham số dòng lệnh nào để tuỳ chỉnh, mọi threshold đều hard-code):
1. Cài dependency, xác thực HuggingFace, load dataset
2. Load/cache model YOLO qua HuggingFace Hub dataset repo (`{username}/mot17-botsort-results`)
3. Chạy detect + track (`model.track(...)`) trên toàn bộ 600 frame
4. Lọc false-positive 2 giai đoạn, lưu kết quả dạng MOTChallenge (`frame,id,x,y,w,h,score,-1,-1,-1`) vào `tracking_results/MOT17-02-FRCNN.txt`
5. Render video có annotate bounding box + ID (`output_tracking.mp4`)
6. Đánh giá bằng TrackEval (fallback `motmetrics` nếu TrackEval lỗi), ghi kết quả vào `eval_results.txt`
7. Upload prediction file, video, và kết quả đánh giá lên HuggingFace Hub dataset repo

## Kết quả đánh giá

**Số lượng box qua từng giai đoạn lọc:**

| Giai đoạn | Số box | Ghi chú |
|---|---|---|
| Raw tracking output (trước lọc) | 6,551 | Output trực tiếp từ YOLOv8 + BoT-SORT |
| Sau lọc tự động (giai đoạn 1) | 6,317 | Loại 4 track ID tĩnh (25, 143, 174, 181) — 234 box |
| Sau lọc thủ công (giai đoạn 2) | 6,188 | Loại thêm 129 box từ 6 ID xác nhận (44, 93, 99, 169, 196, 288) + phần đầu tĩnh của ID 80 (loại đến frame 250) |

Tổng cộng loại 363 box (~5.5% raw detections) qua cả 2 giai đoạn.

**Kết quả TrackEval trên tracking output đã lọc (6,188 box):**

| Metric | Giá trị |
|---|---|
| HOTA | 32.07% |
| MOTA | 27.25% |
| IDF1 | 36.66% |

## Cấu trúc thư mục

Repo hiện chỉ gồm 1 file notebook (`52300102_52300107.ipynb`) và bản export `.py` tương ứng (`52300102_52300107.py`) — không có cấu trúc thư mục con nào khác trong repo. Các thư mục dưới đây được code tạo ra khi chạy (trong `/kaggle/working`), không phải cấu trúc có sẵn:

```
/kaggle/working/                        # WORK_DIR — tạo khi chạy
├── hf_home/                            # cache HuggingFace (HF_HOME)
├── hf_cache/                           # cache model YOLO (HUGGINGFACE_HUB_CACHE)
├── ultralytics_config/                 # YOLO_CONFIG_DIR
├── matplotlib_config/                  # MPLCONFIGDIR
├── yolov8m.pt                          # model weight (LOCAL_MODEL_PATH)
├── tracking_results/
│   └── MOT17-02-FRCNN.txt              # PRED_PATH — kết quả tracking dạng MOTChallenge
├── output_tracking.mp4                 # VIDEO_PATH — video annotate
├── trackeval_data/                     # tạo trong run_trackeval(): gt/, trackers/, seqmaps/, output/
└── eval_results.txt                    # EVAL_PATH — kết quả HOTA/MOTA/IDF1
```

[CẦN BỔ SUNG] Cấu trúc dataset input (`/kaggle/input/datasets/duyl8787/mot17-02-frcnn/MOT17-02-FRCNN/{img1,gt}`) là dataset Kaggle bên ngoài, không thuộc repo này.

## License

[CẦN BỔ SUNG] — Không có file license trong repo. Lưu ý: dataset MOT17 có điều khoản sử dụng riêng (không thuộc phạm vi license của code này); model YOLOv8 (Ultralytics) sử dụng license AGPL-3.0 hoặc license thương mại riêng của Ultralytics — cần kiểm tra lại điều khoản áp dụng cho mục đích sử dụng cụ thể.
