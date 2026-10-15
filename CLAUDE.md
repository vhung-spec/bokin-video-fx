# Bokin FX: ghi chú cho mỗi phiên làm việc

App Windows (máy cá nhân, cho nhân viên) thêm hiệu ứng vào video loop: tuyết, lá, mưa, hạt sáng, video/ảnh ngoài;
vùng đứng yên; vùng riêng cho từng hiệu ứng; tiền cảnh che hiệu ứng. Cách dùng: `README.md`.

## Repo này công khai

- Không đưa vào đây bất cứ gì từ các repo riêng tư của công ty: mã, tài liệu, thông tin nội bộ, tên miền, email, IP,
  token. Đọc chúng chỉ để hiểu cách làm, rồi tự viết. Cũng không nêu tên hay chi tiết của chúng ở đây.
- Không commit khoá, mật khẩu, token. App tự cập nhật từ Releases công khai, không cần khoá.
- Bản cài trên Releases ai cũng tải được. Muốn giới hạn người dùng thì phải làm bản web có đăng nhập (chưa làm).
- Hình hạt (lá, tuyết…) đều vẽ bằng mã. Chỉ thêm hình/video mà công ty sở hữu hoặc có giấy phép.

## Cấu trúc

- `bokin_fx.py`: điểm vào (`--selftest`, `--close`, `--updated`).
- `fx/app.py`: vỏ app trên máy. Cổng 7890–7899, chỉ chạy một bản. Cửa sổ Edge/Chrome `--app`. Tự thoát 2 phút sau khi
  đóng cửa sổ (nếu không có việc), cài bản cập nhật lúc thoát.
- `fx/server.py`: máy chủ `ThreadingHTTPServer`, API JSON `/api/...`. Chỉ nhận Host 127.0.0.1/localhost. Mọi POST cần
  header `X-Bokin: 1`. Trang không bao giờ đưa đường dẫn: chỉ có mã dự án/lớp, kiểm theo mẫu.
- `fx/index.html`: toàn bộ giao diện (JavaScript thuần, không framework, không bước build), tiếng Việt.
- `fx/projects.py`: dự án trong `%APPDATA%\Bokin\FX\projects\<id>` (project.json, input.*, still.png, masks/, files/).
- `fx/engine.py`: ghép một khung (video → vùng đứng yên → các lớp theo thứ tự), kết xuất, xem một khung.
- `fx/effects.py`: tuyết, lá, mưa, hạt sáng; `SCHEMA` là nguồn duy nhất của các thông số (trang đọc từ `/api/schema`).
- `fx/sprites.py`: hình hạt vẽ bằng mã. `fx/overlay.py`: lớp từ video/ảnh ngoài, nối loop bằng hoà mượt.
- `fx/video.py`: ffmpeg (bản đi kèm imageio-ffmpeg): đọc khung, ghi H.264, nối dài bằng stream copy.
- `fx/jobs.py`: hàng đợi xem thử và xuất video. `fx/updater.py`: tự cập nhật. `fx/selftest.py`: tự kiểm tra.
- `packaging/`: PyInstaller (`BokinFX.spec`), Inno Setup (`BokinFX.iss`), icon, bài thử cài đặt và cập nhật trong CI.
- Thư viện: numpy, Pillow, imageio-ffmpeg. **Không dùng OpenCV**: OpenCV cần Media Foundation, mà Windows bản N và
  Windows Server không có sẵn.

## Quy tắc phải giữ

- **Loop liền mạch**: mọi chuyển động là hàm của `t / P` (P = số khung của loop), mọi tần số là số nguyên lần mỗi vòng.
  Vì vậy khung P trùng khung 0. Hạt rơi đi hết khung đúng k lần mỗi vòng; tốc độ vẫn tự do vì đường đi có thể dài hơn
  khung. Video ngoài thì hoà mượt đoạn cuối vào đoạn đầu. Thêm hiệu ứng mới thì thêm vào bài kiểm thử
  `tests/test_effects.py` (khung P == khung 0).
- **Vùng đứng yên giống hệt từng điểm ảnh** của ảnh gốc / khung đã chọn (kiểm trong selftest và tests).
- **Video ra**: H.264 4:2:0, BT.709, đúng kích thước và số khung/giây của video vào, không âm thanh, **không còn gì
  nêu tên công cụ** (mã hoá vào file tạm rồi remux sạch thẻ, bỏ SEI của x264). Video dài: nối bằng stream copy.
- Hiệu ứng nhẹ, tự nhiên. Không bẻ cong cả bức ảnh.
- File tải lên là không tin cậy: chỉ đọc bằng mã của mình, chỉ ghi vào thư mục dữ liệu, giới hạn kích thước.
- Mã và chú thích trong mã bằng tiếng Anh; chữ trên trang bằng tiếng Việt, câu dễ hiểu cho người không rành kỹ thuật.

## Kiểm tra và phát hành

- Nhanh: `python -m unittest discover -s tests -t .`. Trọn vẹn: `python bokin_fx.py --selftest`.
- Nhánh mặc định và nhánh phát hành: `claude/relaxed-ptolemy-v732bs`. Push lên đó thì workflow
  `.github/workflows/bokin-fx.yml` chạy trên Windows:
  1. Kiểm thử, rồi build PyInstaller.
  2. Bản đã build chạy `--selftest`, rồi build bộ cài Inno Setup.
  3. Cài thật, cài đè lúc app đang chạy, rồi thử tự cập nhật lên bản `N.1` với một GitHub giả.
  4. Phát hành Release `fx-v1.0.<số lần chạy>` (ghi SHA-256 vào ghi chú).
- Báo xong chỉ khi CI xanh. Báo đúng kết quả, kể cả khi lỗi.

## Làm việc với chủ dự án

- Trả lời tiếng Việt, câu ngắn, từng bước bấm cụ thể. Chủ dùng Windows, không phải lập trình viên.
- Không xin mật khẩu, khoá, token qua chat. Nhắc che chúng khi chụp màn hình.
- Không tạo pull request nếu chủ không yêu cầu.

## Việc có thể làm sau

- Bản web trên máy chủ của công ty. `server.App(desktop=False)` đã tắt các việc chỉ hợp với máy cá nhân. Còn thiếu:
  biết ai đang dùng (đăng nhập), thư mục riêng cho từng người, giới hạn.
- Tự nhận diện vùng (bấm vào vật để khoanh). Phải dùng mô hình có giấy phép thương mại.
- Kết xuất nhanh hơn: chia khung cho nhiều tiến trình.
