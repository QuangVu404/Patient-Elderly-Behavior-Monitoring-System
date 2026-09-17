# Training trên Kaggle

## Chỉ gắn Kaggle Input, không tải dataset

Người chạy **không tải file ZIP/video của năm dataset về máy**, không giải nén và không
upload lại. Thao tác duy nhất với raw data là chọn **Kaggle Notebook -> Add Input**. Kaggle
đưa nguồn đã có sẵn vào `/kaggle/input`; notebook chỉ đọc tại đó và không copy raw data
sang `/kaggle/working`.

Không có lệnh `kaggle datasets download`, KaggleHub, Hugging Face, HTTP, `wget` hay `curl`
trong pipeline. Kaggle cũng không cho code Python gắn thêm Input vào một session đã chạy,
do đó phải bấm **Add Input trước khi Run All**. Nếu thiếu nguồn, code dừng và báo slug cần
gắn thay vì tự lấy dữ liệu.

Lưu ý: cell `%pip install` chỉ cài thư viện Python, không tải dataset. Nếu notebook bật
Internet thì pip có thể truy cập PyPI. Điều này độc lập với raw data; có thể bỏ cell đó nếu
Kaggle image đã có đúng phiên bản thư viện.

Pipeline mặc định dùng profile `kaggle_rgb_5`: năm nguồn có thể gắn trực tiếp bằng
**Add Input**, không cần tải raw data về máy và không cần upload private dataset lớn.
Classifier `MultiScaleTemporalGRU` được huấn luyện từ đầu trên chuỗi 17 keypoint; YOLO-Pose
chỉ phát hiện người/keypoint.

## Năm nguồn có sẵn để gắn trực tiếp trên Kaggle

| Family trong code | Kaggle slug | Dung lượng gần đúng | Cách dùng |
|---|---|---:|---|
| `FallVision` | [`payutch/fall-video-dataset`](https://www.kaggle.com/datasets/payutch/fall-video-dataset) | 15,36 GB | Mirror tổng hợp; adapter bỏ representation processed và phát cảnh báo phần dư |
| `CAUCAFall` | [`tuyenldvn/caucafall`](https://www.kaggle.com/datasets/tuyenldvn/caucafall) | khoảng 7,75 GB | 10 subject, 5 fall + 5 ADL; đọc PNG trực tiếp |
| `URFD` | [`shahliza27/ur-fall-detection-dataset`](https://www.kaggle.com/datasets/shahliza27/ur-fall-detection-dataset) | 4,18 GB | Chuỗi ảnh RGB fall/ADL |
| `MCFD` | [`soumicksarker/multiple-cameras-fall-dataset`](https://www.kaggle.com/datasets/soumicksarker/multiple-cameras-fall-dataset) | khoảng 3,5 GB | 24 scenario x 8 camera; benchmark đa góc nhìn |
| `UCF101` | [`matthewjansen/ucf101-action-recognition`](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition) | 6,74 GB | Chỉ hard-negative |

Sau khi Add Input, panel Kaggle hiển thị title của từng Dataset. Tùy phiên bản Kaggle,
filesystem dùng schema cũ `/kaggle/input/<slug>/` hoặc schema mới
`/kaggle/input/datasets/<owner>/<slug>/`. Code hỗ trợ cả hai và còn nhận diện theo cấu trúc
thực tế bên trong:

Stage 00 không chỉ kiểm tra tên mount. Nó dừng nếu thiếu inventory canonical: FallVision
5.866 raw video, CAUCAFall 100 sequence, URFD 70 independent sequence, MCFD 192 video và
UCF101 13.320 clip. File bổ sung của uploader tạo `PROVENANCE WARNING` thay vì làm notebook
dừng; cảnh báo được lưu trong `provenance_audit.json`. UCF101 được lọc theo tên canonical và
deduplicate giữa các thư mục split.

```text
/kaggle/input/datasets/<owner>/<slug>/
+-- Fall Video Dataset (chỉ lấy cây FallVision raw):
|   +-- Fall/
|   +-- No_Fall/
+-- Multiple Cameras Fall Dataset:
|   +-- dataset/chute01..chute24/cam1..cam8.avi
+-- ur fall detection dataset:
|   +-- UR_fall_detection_dataset_.../
+-- CAUCAFall:
|   +-- 10 subject folders/10 activity folders/*.png
+-- UCF101 - Action Recognition:
    +-- train/
    +-- test/
    +-- val/
    +-- train.csv, test.csv, val.csv
```

Không cần chuyển các thư mục con ra ngoài, đổi tên hoặc gộp chúng. Adapter quét đệ quy từ
mount tương ứng: `No_Fall` được chuẩn hóa thành non-fall, MCFD được group theo scenario,
URFD giữ dạng chuỗi ảnh, và UCF101 đọc video trong cả ba split directory.

Tổng raw input khoảng 38 GB nhưng `/kaggle/input` là read-only mount, không bị tính vào giới
hạn 20 GB output của `/kaggle/working`. Không chọn dataset thương mại "10,000 videos": trang
Kaggle chỉ chứa preview nhỏ. Dataset re-upload có thể ghi license khác nguồn gốc; khi công bố
kết quả vẫn phải kiểm tra, trích dẫn và tuân thủ license gốc.

Nguồn bổ sung tùy chọn cho gold pose:
[`simuletic/cctv-incident-dataset-fall-and-lying-down-detection`](https://www.kaggle.com/datasets/simuletic/cctv-incident-dataset-fall-and-lying-down-detection).
Nó có ảnh synthetic và COCO-17 keypoint, chỉ dùng trong stage 01 sau khi tạo
`pose_dataset.yaml`, không dùng thay video temporal.


## Chuẩn bị code Input

Kaggle không tự thấy file trong Git/local. Tạo một private Kaggle Dataset nhỏ chứa:

```text
eldercare-training-code/
+-- kaggle_staged.py
+-- pose_dataset.example.yaml
+-- yolov8n-pose.pt
```

Đây là bundle mã nguồn nhỏ, không chứa bất kỳ raw video/ảnh training nào. Tạo một private
Kaggle Dataset từ ba file có sẵn trong dự án rồi gắn nó vào các notebook. Không đưa thư mục
raw data vào bundle này.

`yolov8n-pose.pt` là checkpoint khởi tạo nhỏ, không phải raw dataset. Đặt nó trong code
Dataset để Ultralytics không tự tải model từ Internet. Mỗi lần sửa `kaggle_staged.py`, tạo
version mới cho code Dataset và chọn đúng version đó ở tất cả notebook.

Tóm lại, những việc **không cần làm**: tải năm dataset về máy, giải nén chúng, đổi cấu trúc
thư mục, upload lại raw data, hoặc gọi API tải dữ liệu trong notebook.

## Các adapter tự động

- `FallVision`: đọc raw video trong class `Fall`/`No_Fall`, bỏ bản masked/landmarked. Vì mirror
  tổng hợp hiện có nhiều hơn 5.866 raw video canonical, phần dư vẫn là weak-source data và được
  ghi trong `provenance_audit.json`, không được tuyên bố là FallVision canonical.
- `CAUCAFall`: đọc trực tiếp PNG và nhãn `.txt` cùng tên (`0=nofall`, `1=fall`) ở 23 FPS;
  không mở file AVI. Mọi activity cùng subject giữ chung group.
- `URFD`: đọc trực tiếp thư mục `fall-XX-camN-rgb` và `adl-XX-camN-rgb`; không cần ghép MP4.
- `MCFD`: nhận `chute01..24/cam1..8.avi`, group cả tám camera theo scenario; 01-22 là
  weak-positive và 23-24 là confounding/non-fall. Một group non-fall được giữ trong train để model
  học miền hình ảnh MCFD; group còn lại là test độc lập. Validation dùng negative từ nguồn khác.
- `UCF101`: toàn bộ là non-fall; stage 06 lấy tối đa 4.000 source ở 10 FPS để tránh lấn át.

Nếu layout trên Kaggle thay đổi, tạo `eldercare_manifest.csv` theo mẫu
[`eldercare_manifest.example.csv`](eldercare_manifest.example.csv). Manifest thủ công được
ưu tiên vì discovery gộp nó trước rồi loại path trùng.

## Chạy từ A đến Z

### Chạy gộp một notebook để kiểm tra demo

Dùng [`all_in_one_demo_kaggle.ipynb`](all_in_one_demo_kaggle.ipynb) khi cần chứng minh
toàn bộ pipeline chạy end-to-end trong một session. Add Input:

1. Code Dataset chứa `kaggle_staged.py` và `yolov8n-pose.pt`.
2. Cả năm Kaggle Dataset trong bảng trên.
3. Tùy chọn: gold pose Dataset cùng `pose_dataset.yaml`.

Notebook tự truyền artifact trong `/kaggle/working` theo chuỗi:

```text
manifest -> YOLO-Pose -> 5 pose cache -> MultiScaleTemporalGRU -> ONNX/OpenVINO
```

Cấu hình ổn định mặc định đặt `RUN_POSE_FINETUNE=False`: dùng trực tiếp checkpoint pose đã
mount, sau đó cache 30 source cho mỗi nguồn fall, 80 source UCF101 ở 5 FPS/320 px và train
classifier 3 epoch. Cách này kiểm tra end-to-end classifier tự xây dựng nhưng không tuyên bố
đã fine-tune pose. Mọi DataLoader dùng `workers=0`; export ONNX/OpenVINO được bỏ qua để giảm
RAM và checkpoint PyTorch cuối vẫn được tạo. Notebook dừng ngay nếu Kaggle không cấp CUDA.
Khi áp dụng `max_sources`, code lấy mẫu theo từng cặp `split/label` (tối đa ba source tối thiểu
mỗi strata trước khi lấy phần còn lại), tránh validation hoặc test vô tình mất một class.

Có thể đặt `RUN_POSE_FINETUNE=True` để thêm 40 pseudo frame/family và train YOLO 2 epoch ở
416 px/batch 4, nhưng nên chạy phần này bằng notebook stage 01 riêng.

Cell discovery/manifest đầu tiên chỉ đọc filesystem và metadata nên GPU 0% ở đoạn này là
bình thường. GPU bắt đầu hoạt động từ cell `stage01_finetune_pose`. Đây là smoke test có giới
hạn, không thay cho lần huấn luyện đầy đủ. Nếu session dừng giữa chừng thì artifact trong
`/kaggle/working` chưa Save Version sẽ mất; vì vậy full-data vẫn nên chạy các stage 00-07.

### 00 - Discovery và split (CPU)

Mở [`00_build_manifest_kaggle.ipynb`](00_build_manifest_kaggle.ipynb), Add Input:

1. Code Dataset chứa `kaggle_staged.py`.
2. Cả năm Kaggle Dataset trong bảng trên.

Chạy cell inventory trước:

```python
from pathlib import Path

for item in sorted(Path("/kaggle/input").iterdir()):
    print(item)
```

Sau đó **Save Version -> Save & Run All**. Output bắt buộc:

```text
eldercare_manifest/master_manifest.csv
eldercare_manifest/split_report.json
eldercare_manifest/provenance_audit.json
```

Đọc log và xác nhận đủ `FallVision`, `CAUCAFall`, `URFD`, `MCFD`, `UCF101`. Nếu báo
`Required Kaggle datasets are not mounted`, gắn các slug còn thiếu được liệt kê ngay trong
thông báo rồi chạy lại.

### 01 - Fine-tune YOLO-Pose (GPU)

Mở [`01_finetune_pose_kaggle.ipynb`](01_finetune_pose_kaggle.ipynb), Add Input:

- Code Dataset.
- Output notebook 00.
- Cả năm raw dataset để tạo pseudo pose. Stage này cố ý fail nếu thiếu bất kỳ Input nào,
  tránh việc âm thầm huấn luyện trên subset khác với manifest.
- File `yolov8n-pose.pt` trong Code Dataset; code chỉ mở checkpoint đã mount.
- Tùy chọn: Dataset COCO-17 và `pose_dataset.yaml` đã sửa đúng đường dẫn Kaggle.

Mặc định `POSE_MODE='auto'`: dùng pseudo-only nếu không thấy gold YAML, hoặc mixed nếu có.
Pseudo-only là bước bootstrap/domain adaptation từ teacher hiện có, không được xem là
fine-tune có giám sát độc lập. Muốn fine-tune YOLO-Pose có giám sát và đánh giá mAP đáng tin
cậy, hãy gắn gold COCO-17 và đặt `POSE_MODE='mixed'` hoặc `'gold'`.
Output quan trọng:

```text
yolov8n-pose-finetuned.pt
pose_model.sha256
pose_metadata.json
```

Mọi cache stage phải dùng đúng cùng một version output 01.

Notebook 01 full dùng chiến lược teacher-student hai pha, không áp dụng cho all-in-one:

1. Teacher chỉ giữ pseudo label có detection confidence từ `0.75`, keypoint confidence từ
   `0.35` và ít nhất 10/17 keypoint nhìn thấy; lấy tối đa 750 frame mỗi family.
2. Warm-up 8 epoch với 10 layer đầu đóng băng, AdamW `lr=2e-3`; augmentation hình học vừa.
3. Nạp checkpoint tốt nhất, mở toàn bộ model và train sâu thêm 12 epoch với `lr=3e-4`, giảm
   mosaic/augmentation để tinh chỉnh ổn định.
4. Validation pseudo được chia theo group riêng trong từng dataset family, tránh family nhỏ bị mất
   khỏi validation do lấy mẫu toàn cục; mỗi family/split phải đạt số label tối thiểu.
5. So sánh fitness của warm-up và unfreeze; nếu pha unfreeze kém hơn, pipeline tự quay lại best
   checkpoint của warm-up thay vì luôn giữ model cuối.
6. `pose_metadata.json` lưu fitness hai pha, pha được chọn và toàn bộ ngưỡng để tái lập lần chạy.

Khi có gold pose, pseudo label chỉ bổ sung tập train; validation dùng hoàn toàn gold để việc chọn
checkpoint không bị teacher tự chấm nhãn của chính nó.

Vì không có gold ground truth, đây vẫn là self-training có kiểm soát. Không dùng pseudo
validation metric để tuyên bố pose accuracy cải thiện; hiệu quả cuối phải được kiểm tra bằng
metric classifier stage 07 và video độc lập.

Cấu hình stage 01 mặc định ưu tiên ổn định Kaggle: 416 px, batch 8, `workers=0`, không tạo
plot và không export OpenVINO ở stage này. 750 frame/family, 8 epoch warm-up + 12 epoch
unfreeze là full run;
smoke test nên dùng 40 frame/family và 2 epoch như notebook all-in-one.

### 02-06 - Tạo keypoint cache (GPU)

Mỗi notebook chỉ gắn code, output 00, output 01 và đúng một raw dataset:

| Stage | Notebook | Raw Input |
|---:|---|---|
| 02 | [`02_cache_fallvision_kaggle.ipynb`](02_cache_fallvision_kaggle.ipynb) | `payutch/fall-video-dataset` (chỉ raw FallVision) |
| 03 | [`03_cache_caucafall_kaggle.ipynb`](03_cache_caucafall_kaggle.ipynb) | `tuyenldvn/caucafall` |
| 04 | [`04_cache_urfd_kaggle.ipynb`](04_cache_urfd_kaggle.ipynb) | `shahliza27/ur-fall-detection-dataset` |
| 05 | [`05_cache_mcfd_kaggle.ipynb`](05_cache_mcfd_kaggle.ipynb) | `soumicksarker/multiple-cameras-fall-dataset` |
| 06 | [`06_cache_ucf101_kaggle.ipynb`](06_cache_ucf101_kaggle.ipynb) | `matthewjansen/ucf101-action-recognition` |

Save Version từng notebook. Mỗi output chứa `shard_manifest.csv`, `shard_metadata.json`,
`cache_quality_skips.csv` và `pose_cache/*.npz`. Bộ lấy mẫu luân phiên khoảng cách frame để đạt
đúng FPS trung bình yêu cầu, thay vì làm tròn thành 12,5 FPS cho nguồn 25 FPS. Source không đủ 32
pose frame hoặc detection coverage dưới 10% được ghi vào báo cáo và không đưa vào manifest cache.
Raw frame/video không được copy sang `/kaggle/working`.

### 07 - Train classifier (GPU)

Mở [`07_train_temporal_kaggle.ipynb`](07_train_temporal_kaggle.ipynb), Add Input:

- Code Dataset.
- Output của notebook 01 để đóng gói đúng pose checkpoint triển khai.
- Output của đủ năm notebook 02-06.

Không cần raw video hoặc output 00. Stage bắt buộc đủ cả năm family, rồi kiểm tra
SHA của pose model, feature dimension, sample FPS, source trùng và group leakage trước khi
train. Output cuối gồm checkpoint PyTorch, ONNX/OpenVINO, metrics và prediction CSV.
Mặc định giới hạn 10.000 window cho mỗi cặp family/split, chia ngân sách đều theo các nhãn có sẵn
rồi reservoir sampling từng stratum. Cách này giữ RAM ổn định mà không để lớp chiếm đa số loại
mất lớp thiểu số. Pipeline dùng `workers=0` để ổn định trên Kaggle.

Classifier mới kết hợp temporal convolution ở ba dilation `1/2/4`, bidirectional GRU,
attention pooling và max pooling. Quá trình train dùng feature/frame masking, label smoothing,
gradient clipping và cosine learning-rate schedule. Checkpoint ghi `model_name` để công cụ
evaluate/export vẫn đọc được cả model cũ lẫn model mới.

Pipeline chọn epoch bằng Average Precision, không phụ thuộc threshold và ổn định hơn khi xác suất
chưa được calibration. Sau khi khóa epoch tốt nhất, pipeline tìm đồng thời threshold và số dự đoán
liên tiếp (mặc định 3, 4 hoặc 5) trên validation. Chính sách được chọn phải ưu tiên giữ event recall
tối thiểu 0,85 và false alarms/hour không quá 5; nếu validation không có phương án đạt cả hai,
`selection_status` ghi rõ ràng ràng buộc nào chưa đạt thay vì âm thầm chọn một cấu hình kém an toàn.
Cooldown vẫn là 15 giây giống runtime. Sau training chính, ba epoch learning rate thấp ưu tiên toàn bộ
window thuộc 20% video non-fall gây xác suất fall cao nhất; checkpoint chỉ được thay nếu validation AP
tốt hơn. Metric source/group riêng cho từng dataset nằm trong `per_dataset_aggregates`.

Bộ chọn policy có thêm `recall_safety_margin=0.05`: mặc định yêu cầu pooled validation event recall
ít nhất 0,90. Đồng thời, recall của từng dataset và ba nhóm stress validation phải ít nhất 0,85.
Các nhóm stress được chia luân phiên theo group bên trong từng dataset, không tách một group sang
nhiều nhóm; chỉ nhóm có ít nhất năm source fall mới tham gia ràng buộc recall. Metric của mọi nhóm,
kể cả nhóm nhỏ, nằm trong `training_strategy.operational_policy.validation_cohort_metrics`.
Đây là kiểm tra độ ổn định trên validation của một checkpoint, không phải cross-validation train
lại nhiều model; không dùng test để chọn threshold. Nếu không đạt ràng buộc, pipeline vẫn xuất
candidate và in `POLICY WARNING`, không xác nhận model đã sẵn sàng triển khai.

Để áp dụng thay đổi này, cập nhật Code Dataset trên Kaggle, khởi động lại session rồi chạy lại 07
với output 01 và 02-06 hiện có. Không cần tạo lại pose/cache nếu chúng không đổi. Giữ output cũ
để so event recall, số false alerts và thời lượng negative monitoring, không chỉ accuracy/F1.

Metric theo dataset gồm accuracy, balanced accuracy, precision, recall, F1, specificity và false
positive rate. Với UCF101 chỉ chứa non-fall, hãy đọc specificity/FPR; F1 của riêng nguồn này không
có ý nghĩa đánh giá đủ hai lớp.

`operational_test` báo source recall, specificity, `alert_precision`, `labeled_event_recall` và
`false_alarms_per_hour` sau khi áp dụng confirmation/cooldown. Source prediction chỉ phụ thuộc vào
alert model đã phát, không sử dụng ground-truth tại thời điểm alert. Đây là metric chính để chọn
threshold triển khai; metric dùng max probability trên cả source/group chỉ giữ vai trò chẩn đoán
các nguồn có spike.
Phép đo vận hành chạy cửa sổ dày với stride 1 giống runtime; kết quả chi tiết nằm trong
`operational_test_predictions.csv`. Toàn bộ cấu hình ứng viên trên validation được ghi vào
`operational_policy_audit.csv`; metric vận hành tách theo dataset nằm ở `per_dataset_operational`
trong `metrics.json`.

Ngoài metric theo window, `metrics.json` còn có metric theo source và theo cặp group/label để giảm
ảo tưởng do các cửa sổ chồng lấn. `test_predictions.csv` chứa thêm thời điểm giữa cửa sổ.
`deployment_manifest.json` ghi pose SHA, tên model, sequence length, sample FPS và đúng threshold
được chọn. Khi dùng OpenVINO, runtime đợi đủ cửa sổ 32 frame thay vì padding từ 12 frame.
Runtime cũng lấy mẫu theo đúng `sample_fps` 10 FPS đã ghi trong manifest, kể cả khi camera/video
gốc chạy ở FPS cao hơn.
`windowing_audit.json` liệt kê rõ cache còn ngắn hơn sequence length nếu người dùng tăng độ dài
cửa sổ ở stage 07.

Sau khi tải nguyên thư mục output 07 về, tạo config chạy thật mà không phải chép model/threshold
thủ công:

```powershell
python tools/install_training_bundle.py path/to/eldercare_training
eldercare-monitor --config configs/trained.yaml --source 0
```

Nếu output nằm ngay trong repository như cấu trúc hiện tại, dùng:

```powershell
python tools/install_training_bundle.py training/eldercare_training --classifier-backend pytorch --force
streamlit run streamlit_app.py
```

Giao diện Streamlit tự chọn `configs/trained.yaml`; tên config đang hoạt động được hiển thị ở
sidebar. Backend PyTorch dùng được ngay với môi trường dự án. Muốn giảm latency trên CPU, cài
`pip install -e ".[openvino]"` rồi tạo lại config với `--classifier-backend openvino`.

## Manifest thủ công

Các cột bắt buộc là `dataset,path,label,group`. `path` nên tương đối với file manifest.

```csv
dataset,path,label,group,source_type,segments
FallVision,Fall/Raw_Video/video_001.mp4,1,subject_01,video,
FallVision,No_Fall/Raw_Video/video_002.mp4,0,subject_02,video,
MCFD,dataset/chute01/cam1.avi,1,mcfd_scenario_01,video,
URFD,UR_fall_detection_dataset_cam0_rgb/fall-01-cam0-rgb,1,urfd_fall_01,images,
```

Quy tắc:

- `label`: `0` non-fall, `1` fall, `-2` nếu dùng `segments`.
- `segments`: JSON `[[label,start_second,end_second], ...]`.
- Mọi camera/clip của cùng subject hoặc original recording phải dùng cùng `group`.
- Tên `dataset` phải thuộc đúng năm family; alias thông dụng được chuẩn hóa tự động.

## Kiểm tra và xử lý lỗi

```python
from pathlib import Path
import pandas as pd

master = next(Path("/kaggle/input").rglob("master_manifest.csv"))
df = pd.read_csv(master)
print(df.groupby(["dataset", "split", "label"]).size())
print("missing paths:", sum(not Path(p).exists() for p in df.path))
```

| Lỗi | Nguyên nhân/cách xử lý |
|---|---|
| `Required Kaggle datasets are not mounted` | Chưa Add Input; gắn các slug được liệt kê trong thông báo rồi chạy lại |
| `Dataset provenance/layout audit failed` | Mirror thiếu dữ liệu canonical; đối chiếu số lượng/cấu trúc trong `datasets_catalog.md` |
| `[PROVENANCE WARNING]` | Mirror có file bổ sung; pipeline vẫn chạy nhưng không được xem phần dư là dữ liệu canonical độc lập |
| FFmpeg báo `ac-tex damaged`/`Error at MB` | Một số macroblock AVI lỗi; code mới tắt log lặp, bỏ frame hỏng và cắt riêng video sau quá 32 lỗi |
| `Stage 07 is missing cache families` | Chưa gắn đủ output của năm notebook 02-06 |
| CAUCAFall không được nhận | Kiểm tra slug `tuyenldvn/caucafall` và các activity folder còn chứa PNG |
| URFD không được nhận | Cần giữ nguyên tên thư mục `fall/adl-XX-camN-rgb` |
| MCFD không được nhận | Cần có `chute01..chute24`, bên trong là `cam1.avi..cam8.avi` |
| `Raw source is not attached` | Cache notebook chưa gắn đúng raw dataset/version đã dùng ở stage 00 |
| Pose SHA mismatch | Các cache notebook đang dùng khác version output 01 |
| Train thiếu một class | Kiểm tra FallVision/CAUCAFall/URFD có cả fall và non-fall trong các split |
| Hết thời gian | Hạ cùng một `sample_fps` cho cả năm cache stage, hoặc hạ `image_size`/`max_sources`; không copy raw input |

## Giới hạn đánh giá

Đây là cấu hình đủ để xây dựng và chứng minh pipeline cá nhân, không phải chứng nhận y tế.
Mirror chứa FallVision là một gói tổng hợp nên adapter chỉ chọn raw `Fall/No_Fall`; không suy
diễn license từ uploader. Nhãn MCFD hiện ở mức sequence, không phải temporal ground truth.
CAUCAFall/URFD/MCFD phải được đối chiếu nguồn gốc; UCF101 chỉ đóng vai trò hard-negative.
Báo cáo metric theo từng dataset và luôn dành một nguồn độc lập làm cross-domain test.
