# Hệ thống giám sát hành vi người cao tuổi/bệnh nhân trên CPU

Starter project cho pipeline thời gian thực:

```text
Camera / RTSP
      v
YOLOv8n-Pose + ByteTrack
      v
Chuẩn hóa 17 keypoint + vận tốc + hình học bbox
      v
Temporal Conv1D + Attention Pooling (hoặc heuristic bootstrap)
      v
Xác nhận nhiều frame + cooldown -> cảnh báo JSONL / snapshot tùy chọn
```

Mục tiêu của mã nguồn là tạo một baseline đo được và dễ thay thế từng phần trên máy chỉ có CPU. Nó nhận diện hai sự kiện: `fall` (nguy cơ ngã) và `immobility` (nằm ngang ít chuyển động quá lâu). Đây **không phải thiết bị y tế**; trước triển khai thật cần kiểm định tại đúng phòng, góc camera và nhóm người sử dụng.

## Những gì đã có

- Nhận webcam, file video hoặc RTSP bằng OpenCV.
- YOLOv8n-Pose và ByteTrack qua `model.track(..., persist=True, tracker="bytetrack.yaml")`.
- Vector đặc trưng 89 chiều/frame: tọa độ chuẩn hóa, confidence, vận tốc keypoint và hình học bbox.
- Cửa sổ temporal tách biệt theo `track_id`, tự dọn track mất quá lâu.
- Mạng rất nhỏ: projection -> depthwise temporal Conv1D -> learned attention pooling -> 2 lớp `normal/fall`.
- Chế độ heuristic để chạy thử khi chưa có checkpoint; không nên dùng làm ngưỡng production.
- Debounce, cooldown, phát hiện nằm bất động theo thời gian thực tế, JSONL audit log và snapshot opt-in.
- Công cụ tạo dataset, train theo group/subject để tránh leakage, xuất OpenVINO FP16/INT8 và benchmark.

Ultralytics xác nhận pose model dùng được với tracking, ByteTrack không dùng ReID và `persist=True` duy trì trạng thái qua các frame liên tiếp trong [tài liệu tracking chính thức](https://docs.ultralytics.com/modes/track/). OpenVINO khuyến nghị bắt đầu bằng PTQ với tập calibration đại diện rồi mới cân nhắc pruning/QAT nếu trade-off chưa đạt; xem [basic quantization flow](https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html) và [model optimization guide](https://docs.openvino.ai/2025/openvino-workflow/model-optimization.html).

## Cài đặt

Yêu cầu Python 3.10+ (khuyến nghị 3.11) và webcam/video thử nghiệm.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Ứng dụng local cần checkpoint `yolov8n-pose.pt` tại đường dẫn `detector.model`. Riêng pipeline
Kaggle không tự lấy checkpoint qua mạng: hãy gắn Code Input đã chứa file này theo
[`training/README.md`](training/README.md).

Nếu đã có output notebook 07 trong `training/eldercare_training`, cài bundle một lần:

```powershell
python tools/install_training_bundle.py training/eldercare_training --classifier-backend pytorch --force
```

Lệnh này tạo `configs/trained.yaml` với đúng YOLO-Pose, classifier đã chọn, độ dài cửa sổ,
sample FPS và threshold từ lần train. Chạy webcam bằng CLI:

```powershell
eldercare-monitor --config configs/trained.yaml --source 0
```

Chạy giao diện Streamlit:

```powershell
streamlit run streamlit_app.py
```

Streamlit tự ưu tiên `configs/trained.yaml` nếu file tồn tại; nếu không, ứng dụng dùng
`configs/default.yaml`. Có thể ép một config khác bằng biến môi trường `ELDERCARE_CONFIG`.
Trong trình duyệt, có thể tải video hoặc nhập `0` cho webcam, đường dẫn video hay URL RTSP.

Cấu hình trên dùng checkpoint PyTorch có sẵn trong output 07 nên chạy được ngay với dependency
cơ bản. Để tối ưu CPU bằng OpenVINO, cài `pip install -e ".[openvino]"`, rồi chạy lại lệnh cài
bundle với `--classifier-backend openvino`.

Chạy video hoặc RTSP:

```powershell
eldercare-monitor --source data/raw/demo.mp4
eldercare-monitor --source "rtsp://user:password@camera/stream" --headless
```

Không ghi URL chứa mật khẩu vào config được commit. Nhấn `q` để thoát giao diện. Sự kiện nằm tại `artifacts/events.jsonl`.

## Dữ liệu và huấn luyện

Pipeline Kaggle nhiều stage dùng profile `kaggle_rgb_5` gồm FallVision, CAUCAFall, URFD,
MCFD và UCF101 nằm trong [`training/`](training/).
Classifier temporal trong notebook là mô hình tự xây dựng và được train từ đầu. YOLO-Pose
vẫn chỉ đảm nhiệm trích 17 keypoint, nhưng có thể fine-tune bằng nhãn pose thật hoặc
pseudo-label lấy riêng từ training groups. Xem hướng dẫn và schema nhãn tại
[`training/README.md`](training/README.md).

Để smoke-test toàn bộ pipeline trong một session, dùng
[`training/all_in_one_demo_kaggle.ipynb`](training/all_in_one_demo_kaggle.ipynb).
Để train đầy đủ năm nguồn và có thể tiếp tục khi một session bị ngắt, dùng bộ notebook stage bắt đầu từ
[`training/00_build_manifest_kaggle.ipynb`](training/00_build_manifest_kaggle.ipynb)
và kết thúc tại
[`training/07_train_temporal_kaggle.ipynb`](training/07_train_temporal_kaggle.ipynb).

Không trộn ngẫu nhiên clip/frame của cùng một người vào cả train và validation. Cột `subject` trong manifest là đơn vị group split. Với UR Fall hoặc video tự quay, tạo manifest theo [mẫu](data/manifest.example.csv):

```csv
video_path,label,start_sec,end_sec,subject
data/raw/local/fall_01.mp4,1,2.0,5.5,person_01
data/raw/local/pick_object.mp4,0,0,12,person_02
```

`label=1` là đoạn ngã, `label=0` là ADL/negative. Hãy cắt interval sát hành động; nếu gán toàn bộ clip dài là fall, nhãn nhiễu sẽ rất lớn. ActivityNet chỉ nên dùng các clip có giấy phép/phân lớp phù hợp; negative quan trọng nhất vẫn là video đúng phòng triển khai: ngủ sofa, nằm giường, ngồi xuống nhanh, cúi nhặt đồ, che khuất và người chăm sóc đi ngang.

Chuẩn bị dữ liệu ở 10 FPS và cửa sổ 32 frame:

```powershell
pip install -e .[train]
python tools/prepare_dataset.py data/manifest.csv --output data/processed/sequences.npz
```

Huấn luyện:

```powershell
python tools/train_classifier.py data/processed/sequences.npz `
  --output artifacts/temporal_attention.pt --epochs 40
```

Script lưu checkpoint tốt nhất theo validation F1, tự quét threshold và ghi metric cùng `recommended_threshold`. Chép ngưỡng được chọn vào config, sau đó khóa ngưỡng và chỉ đánh giá một lần trên test set độc lập.

Đánh giá checkpoint trên tập test hold-out:

```powershell
python tools/evaluate_classifier.py artifacts/temporal_attention.pt `
  data/processed/test_sequences.npz --threshold 0.72
```

FPR từ script là theo cửa sổ, không thay thế phép đo số cảnh báo giả/giờ trên video liên tục.

## Xuất và dùng OpenVINO

Cài phần tối ưu:

```powershell
pip install -e .[train,openvino]
```

Classifier FP16:

```powershell
python tools/export_openvino.py artifacts/temporal_attention.pt `
  --output artifacts/temporal_openvino/model.xml
```

Classifier INT8 PTQ với tối đa 300 mẫu calibration:

```powershell
python tools/export_openvino.py artifacts/temporal_attention.pt `
  --calibration data/processed/sequences.npz `
  --output artifacts/temporal_openvino_int8/model.xml
```

Sau đó đặt trong `configs/default.yaml`:

```yaml
analysis:
  classifier_model: artifacts/temporal_openvino_int8/model.xml
```

Nếu dùng output đầy đủ của Kaggle stage 07, có thể tạo config đã liên kết pose model, classifier,
sequence length và threshold tự động:

```powershell
python tools/install_training_bundle.py path/to/eldercare_training
eldercare-monitor --config configs/trained.yaml --source 0
```

Config sinh ra buộc runtime dùng đủ sequence và đúng sample FPS như lúc tạo cache, tránh lệch miền
thời gian giữa video huấn luyện và camera triển khai.

Xuất pose detector sang OpenVINO:

```powershell
python tools/export_detector.py --model yolov8n-pose.pt --image-size 416
```

Lệnh trả về thư mục model; dùng đường dẫn đó tại `detector.model`. INT8 cần YAML của tập ảnh calibration đại diện:

```powershell
python tools/export_detector.py --int8 --data configs/calibration.yaml
```

Tài liệu tích hợp chính thức của Ultralytics mô tả export/inference OpenVINO tại [Intel OpenVINO export](https://docs.ultralytics.com/integrations/openvino/). INT8 cần được so lại recall/FPR; kích thước nhỏ hơn không tự động bảo đảm nhanh hơn trên mọi CPU.

## Benchmark và tuning CPU

```powershell
python tools/benchmark.py --source data/raw/demo.mp4 --model yolov8n-pose.pt `
  --image-size 416 --frames 300
```

So sánh cùng video, cùng số frame và bỏ warm-up. Ghi lại ít nhất mean latency, p95, FPS, recall fall và false alarms/hour. Tuning theo thứ tự thực dụng:

1. Giảm `image_size` từ 416 xuống 320 và đo mất mát keypoint/recall.
2. Dùng OpenVINO FP16, sau đó INT8 với calibration đúng miền dữ liệu.
3. Nếu vẫn thiếu FPS, tăng `process_every_n_frames` lên 2; thời gian sự kiện vẫn dùng clock thực.
4. Chỉ pruning sau khi PTQ chưa đạt mục tiêu và có quy trình fine-tune/đánh giá lại.

Không khẳng định trước mức "tiết kiệm 90%"; hãy báo cáo tỷ lệ đo trên phần cứng đích so với baseline pixel-video (ví dụ CNN 3D) bằng cùng dữ liệu và tiêu chí.

## Kiểm thử

```powershell
pip install -e .[dev]
pytest -q
ruff check src tools tests
```

Unit test bao phủ tạo đặc trưng, padding cửa sổ, tách track, xác nhận liên tiếp/cooldown và việc bất động dùng wall-clock thay vì độ dài cửa sổ model.

## Tiêu chí nghiệm thu đề xuất

| Nhóm | Chỉ số nên khóa trước khi test |
|---|---|
| An toàn | Recall fall, missed falls theo từng edge case |
| Báo giả | Precision, false alarms/hour, riêng ngủ sofa/cúi nhặt đồ |
| Thời gian | Detection-to-alert latency p50/p95 |
| Hiệu năng | FPS end-to-end, p95 frame latency, CPU/RAM, nhiệt độ sau 8 giờ |
| Tracking | ID switches và thời gian phục hồi sau che khuất |
| Vận hành | Camera mất kết nối, log rotation, watchdog, chính sách lưu/xóa ảnh |

Với production, nên bổ sung heartbeat camera, reconnect có exponential backoff, health endpoint, giới hạn dung lượng log, mã hóa dữ liệu và kênh cảnh báo có retry/acknowledgement. Những phần đó phụ thuộc hạ tầng nhận cảnh báo nên chưa được tự động giả định trong starter này.

## Cấu trúc

```text
configs/default.yaml                 cấu hình runtime
src/eldercare_monitor/detector.py    YOLO Pose + ByteTrack
src/eldercare_monitor/features.py    chuẩn hóa và cửa sổ temporal
src/eldercare_monitor/model.py       Temporal Attention 1D
src/eldercare_monitor/analyzer.py    state machine ngã/bất động
src/eldercare_monitor/pipeline.py    vòng lặp end-to-end
tools/prepare_dataset.py             video manifest -> NPZ
tools/train_classifier.py            train/group validation
tools/evaluate_classifier.py         đánh giá hold-out
tools/export_openvino.py             FP16/INT8 classifier
tools/export_detector.py             OpenVINO pose detector
tools/benchmark.py                   latency/FPS
tests/                               unit tests không cần model
```
