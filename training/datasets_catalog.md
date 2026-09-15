# Catalog dữ liệu cho profile `kaggle_rgb_5`

Pipeline chỉ đọc dữ liệu được gắn bằng **Kaggle Notebook -> Add Input** và không tự tải qua
mạng. Tên trên Kaggle không phải bằng chứng nguồn gốc; mỗi mirror phải khớp DOI/trang phát hành
và cấu trúc canonical dưới đây.

| Family | Input Kaggle | Dấu hiệu nhận dạng | Nguồn gốc |
|---|---|---|---|
| `FallVision` | [`payutch/fall-video-dataset`](https://www.kaggle.com/datasets/payutch/fall-video-dataset) | Đọc raw video trong `Fall` và `No_Fall`; bỏ masked/landmarked/processed; file vượt quá 5.866 canonical được đánh dấu weak-source | [Harvard Dataverse, DOI 10.7910/DVN/75QPKK](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/75QPKK). Mirror là gói tổng hợp nên không mặc nhiên kế thừa nhãn CC0 của uploader |
| `CAUCAFall` | [`tuyenldvn/caucafall`](https://www.kaggle.com/datasets/tuyenldvn/caucafall) | 10 subject x 10 activity; AVI + PNG + TXT; `0=nofall`, `1=fall` | [Mendeley Data v5, DOI 10.17632/7w7fccy7ky.5](https://data.mendeley.com/datasets/7w7fccy7ky/5), CC BY 4.0 |
| `URFD` | [`shahliza27/ur-fall-detection-dataset`](https://www.kaggle.com/datasets/shahliza27/ur-fall-detection-dataset) | 30 `fall-XX-cam0-rgb` + 40 `adl-XX-cam0-rgb` | [University of Rzeszów](https://fenix.ur.edu.pl/~mkepski/ds/uf.html); tuân theo điều khoản nguồn gốc |
| `MCFD` | [`soumicksarker/multiple-cameras-fall-dataset`](https://www.kaggle.com/datasets/soumicksarker/multiple-cameras-fall-dataset) | `chute01`...`chute24`, mỗi scenario có `cam1.avi`...`cam8.avi` | [Trang gốc](https://www.iro.umontreal.ca/~labimage/Dataset/) và [technical report](https://www.iro.umontreal.ca/~labimage/Dataset/technicalReport.pdf) |
| `UCF101` | [`matthewjansen/ucf101-action-recognition`](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition) | 13.320 clip, 101 class; tên `v_<class>_gXX_cXX.avi` | [UCF CRCV](https://www.crcv.ucf.edu/research/data-sets/ucf101/); chỉ dùng làm hard-negative |

## Quy tắc chống sai mirror và leakage

- Không gọi toàn bộ `FallVideo` là FallVision canonical: đây là gói tổng hợp có cả FallVision
  và MCFD. Adapter loại representation processed nhưng không thể chứng minh nguồn của từng file
  dư chỉ từ tên class. Kết quả dùng mirror phải công bố đây là weak-source aggregate; benchmark
  FallVision canonical cần Input riêng được tạo từ bản Harvard Dataverse.
- Không dùng `CCTVAction`: provenance từng clip không đầy đủ, license không rõ và một số lớp có
  clip nhân đôi.
- MCFD split theo `chuteXX`, không theo camera; tám góc quay của cùng sự kiện không được lọt sang
  các split khác nhau.
- CAUCAFall split theo subject; URFD theo sequence; UCF101 theo group `gXX`.
- Không dùng đồng thời raw và landmarked của cùng video như hai mẫu độc lập.
- License trên trang re-upload không thay thế license của nguồn gốc.

## Giới hạn nhãn MCFD

Scenario 01-22 chứa một sự kiện ngã sau các ADL; scenario 23-24 chỉ có confounding events.
Adapter hiện gán weak label ở mức sequence cho baseline. Không báo cáo metric theo frame của MCFD
như ground truth. Temporal localization cần interval annotation đã đối chiếu technical report.
