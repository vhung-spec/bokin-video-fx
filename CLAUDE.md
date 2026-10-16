# Bokin FX: ghi chú cho mỗi phiên làm việc

App Windows cho chủ dự án tự dùng trên máy của mình. Thêm hiệu ứng vào video loop rồi xuất video 30 giây hoặc 1 phút,
vẫn loop liền mạch. Giao diện giống Premiere: thư viện hiệu ứng (file ngoài là chính, hiệu ứng tạo sẵn là phụ),
timeline có track và clip, xem nhanh ngay trong trình duyệt, vùng đứng yên, vùng riêng cho từng clip, tiền cảnh che
hiệu ứng. Cách dùng: `README.md`.

## Repo này công khai

- Không đưa vào đây bất cứ gì từ các repo riêng tư của công ty: mã, tài liệu, thông tin nội bộ, tên miền, email, IP,
  token. Đọc chúng chỉ để hiểu cách làm, rồi tự viết. Cũng không nêu tên hay chi tiết của chúng ở đây.
- Không commit khoá, mật khẩu, token. App tự cập nhật từ Releases công khai, không cần khoá.
- Bản cài trên Releases ai cũng tải được.
- Hình hạt (lá, tuyết…) đều vẽ bằng mã. Chỉ thêm hình/video mà công ty sở hữu hoặc có giấy phép.
- Học ý tưởng từ dự án mở thì được. Mã GPL (Shotcut, Kdenlive, OpenShot…) không chép vào đây. Mã MIT/Apache chỉ dùng
  khi ghi kèm giấy phép.

## Cấu trúc

- `bokin_fx.py`: điểm vào (`--selftest`, `--close`, `--updated`).
- `fx/app.py`: vỏ app trên máy. Cổng 7890–7899, chỉ chạy một bản. Cửa sổ Edge/Chrome `--app`. Tự thoát 5 phút sau khi
  đóng cửa sổ (nếu không có việc), cài bản cập nhật lúc thoát.
- `fx/server.py`: máy chủ `ThreadingHTTPServer`, API JSON `/api/...`, file cho xem nhanh `/media/...`. Chỉ nhận Host
  127.0.0.1/localhost. Mọi POST cần header `X-Bokin: 1`. Trang không bao giờ đưa đường dẫn: chỉ có mã dự án, clip,
  mục thư viện, kiểm theo mẫu.
- `fx/index.html`: toàn bộ giao diện (JavaScript thuần, không framework, không bước build), tiếng Việt. Bốn vùng như
  Premiere: Bảng chỉnh hiệu ứng, Chương trình (xem nhanh), Thư viện/Tạo sẵn/Dự án, Timeline.
- `fx/timeline.py`: luật thời gian của clip (độ dài 30/60 giây, `Span`: bắt đầu, kết thúc, mờ dần). `index.html` có
  bản sao bằng JavaScript (`span`, `weight`, `plan`, `fileTime`): sửa một bên thì sửa cả bên kia.
- `fx/projects.py`: dự án trong `%APPDATA%\Bokin\FX\projects\<id>` (project.json, input.*, proxy.mp4, still.png,
  masks/, clips/). `layers` là các clip; `tracks` từ dưới lên; `upgrade()` chuyển dự án của bản cũ (một lần);
  `replace_video()` đổi video V1 (file mới tên `input-<x>`, bản nhẹ `proxy-<x>`; thời gian clip giữ theo giây).
- `fx/library.py`: thư viện hiệu ứng dùng chung trong `%APPDATA%\Bokin\FX\library\<id>` (file gốc, ảnh nhỏ, bản nhẹ).
- `fx/engine.py`: ghép một khung (video → vùng đứng yên → các clip theo track, mờ dần), kết xuất song song nhiều đoạn,
  khung chính xác cho trang (`Looks`), clip xem nhanh của hiệu ứng tạo sẵn (`fx_clip`), bản nhẹ của video gốc.
- `fx/effects.py`: phần chung của hiệu ứng tạo sẵn (`Canvas`, `Effect`, `Falling`), lá, hạt sáng, và danh sách `KINDS`
  / `SCHEMA` (nguồn duy nhất của các thông số; trang đọc từ `/api/schema`; `group` = tiêu đề nhóm trên trang, `look` =
  kiểu có sẵn đặt nhiều thông số một lúc). `fx/snow.py`, `fx/rain.py`, `fx/glass.py`: tuyết, mưa, nước trên kính, mỗi
  cái một file. `VERSION` của hiệu ứng tăng khi hình vẽ ra đổi (clip xem nhanh làm lại). Hiệu ứng `lens` (nước trên
  kính) đổi cả hình bên dưới: `apply()` cho video xuất và khung chính xác; trang chỉ phát `draw()` (phần phủ lên), nên
  xem nhanh gần đúng và dừng hình thì hiện khung chính xác (`liveIncomplete`).
- `fx/sprites.py`: hình hạt vẽ bằng mã. `fx/overlay.py`: clip từ video/ảnh ngoài (`timeline()` cho clip suốt video,
  `playing()` cho clip ngắn hơn).
- `fx/video.py`: ffmpeg (bản đi kèm imageio-ffmpeg): đọc khung, ghi H.264, nối các đoạn bằng stream copy, bản nhẹ cho
  trình duyệt (H.264, hoặc VP9 WebM khi có nền trong suốt).
- `fx/jobs.py`: hai hàng đợi: xuất video, và chuẩn bị xem nhanh (bản nhẹ, clip hiệu ứng tạo sẵn).
- `fx/updater.py`: tự cập nhật. `fx/selftest.py`: tự kiểm tra.
- `packaging/`: PyInstaller (`BokinFX.spec`), Inno Setup (`BokinFX.iss`), icon, bài thử cài đặt, cập nhật, và giao
  diện trên Edge (`ci_page_test.py`, dùng Playwright) trong CI.
- Thư viện: numpy, Pillow, imageio-ffmpeg. **Không dùng OpenCV**: OpenCV cần Media Foundation, mà Windows bản N và
  Windows Server không có sẵn.

## Xem nhanh trên trang

- Trình duyệt tự ghép: `<video>` bản nhẹ của video gốc, ảnh vùng đứng yên, rồi từng clip với CSS `mix-blend-mode`
  (screen, multiply, plus-lighter cho add), `opacity` (mờ dần) và `mask-image` (vùng đã làm mềm; clip "sau tiền cảnh"
  thêm tiền cảnh đảo ngược, `mask-composite: intersect`). Một đồng hồ chung giữ mọi video khớp nhau.
- Hiệu ứng tạo sẵn: app kết xuất cả độ dài video thành clip nhỏ (WebM trong suốt, hoặc H.264 cho hạt sáng dùng
  Screen). Tên file theo nội dung (`fx-<khoá>.webm`), hai nửa của clip bị cắt dùng chung.
- Khung chính xác từ `engine.Looks` (`/api/look`) là thứ video xuất ra. Trang chỉ hiện nó khi dừng hình nếu người dùng
  bật 🎯, hoặc khi bản xem nhanh còn thiếu (`liveIncomplete`). Không tự thay ảnh khi dừng: mắt thấy như hình bị giật.
- Khi phát, video gốc là đồng hồ; dừng thì giữ khung đang hiện (`requestVideoFrameCallback`). Bản nhẹ đánh số khung
  đều (`setpts=N/(fps*TB)`, `-fps_mode cfr`), nên khung n của trang là khung n của app.
- Vẽ vùng: nét cọ vẽ thẳng lên mặt nạ và lớp màu; việc nặng (gửi vùng, làm mềm cho xem nhanh) chỉ chạy khi ngừng vẽ.
- Khung chỉnh clip (`#xform`, `placed()`): cùng công thức đặt clip với `overlay.Overlay` (và `Effect.box()` của
  hiệu ứng tạo sẵn đặt một chỗ: `BOX`, `placeable()`; clip xem nhanh của nó chỉ là cái hộp, trang đặt vào chỗ).
- Màu của clip file: `overlay.colour_ops` (đường cong tông giữ đen là đen, rồi ma trận màu) và trang dựng đúng các số
  đó thành bộ lọc SVG (`colourOps`, `colourFilter`, `feComponentTransfer` bảng + `feColorMatrix`, sRGB).
- Hoàn tác: `S.hist` giữ timeline và cả lần vẽ vùng (mặt nạ trước/sau, chỉ 8 lần gần nhất). Phóng to hình:
  `zoomPicture` (Ctrl +/-/0/1, `+` `-` khi chuột trên hình); Ctrl + lăn chuột không phóng cả trang.
- Trình duyệt Chromium của Playwright không phát H.264. Edge và Chrome thì phát được (CI thử trên Edge).

## Quy tắc phải giữ

- **Loop liền mạch**: mọi chuyển động là hàm của `t / P`, mọi tần số là số nguyên lần mỗi vòng, nên khung P trùng
  khung 0. Với video xuất ra, P là cả độ dài (30 giây hoặc 1 phút = số nguyên vòng loop của video gốc). Hạt rơi đi hết
  khung đúng k lần mỗi vòng; tốc độ vẫn tự do vì đường đi có thể dài hơn khung. Thêm hiệu ứng mới thì thêm vào bài
  kiểm thử `tests/test_effects.py` (khung P == khung 0).
- **Clip trên timeline**: clip suốt video thì lặp liền mạch (video ngoài hoà mượt đoạn cuối vào đoạn đầu). Clip ngắn
  hơn luôn mờ dần ít nhất `MIN_EDGE_FADE` ở đầu và cuối video. Cắt đôi clip không đổi hình: nửa sau có
  `skip` = chỗ đã phát (kiểm trong `tests/test_engine.py`).
- **Không tự kéo dài clip**: clip mới dài bằng file (ảnh, hiệu ứng tạo sẵn: 5 giây). Đổi 30 giây / 1 phút hay đổi
  video gốc chỉ kéo theo clip suốt video. Người dùng tự nối (Nhân đôi: cùng đoạn file, đặt ngay sau); hai clip cùng
  nguồn nối liền nhau thì bỏ mờ dần ở chỗ nối (`joinFades`).
- **Mix** (chuyển cảnh): `mix` của clip sau = số giây hoà trộn với clip ngay trước trên cùng track. `timeline.mixes()`
  và `Span(..., mix)` (trang: `mixAt`, `span`): clip trước chạy thêm nửa Mix sau chỗ nối, clip sau bắt đầu sớm nửa Mix,
  tổng độ đậm bằng 1; ngoài vùng Mix hình như cũ. File đi vòng khi ra ngoài đoạn của clip. Bản sao, nửa sau khi cắt:
  `mix` = 0. Clip sau chỉ tính đến hết video (clip trước không chạy qua chỗ loop). Hoà trộn thật, không tối đi ở giữa:
  `Scene._crossfade` ghép riêng từng clip lên cùng hình rồi trộn hai hình; trang đặt hai clip trong hộp của track
  (`trackBox`, `isolation: isolate`, chế độ hoà trộn của clip), clip trước `normal`, clip sau `plus-lighter`
  (`mixGroups`); hai clip khác chế độ hoà trộn thì trang xếp lần lượt (gần đúng). Clip mất clip trước (xoá, kéo đi)
  thì bỏ `mix` (`edited`). Bỏ mờ dần ở chỗ nối chỉ khi clip sau chạy tiếp clip trước (`continues`); tách ra thì
  mờ dần lại (`fadesBack`).
- **Phát ngược**: `params.reverse` của clip file. `library.make_reverse()` làm một lần bản phát ngược cỡ thật
  (`video.reverse`: đọc file một lượt, mỗi đoạn ghi ngược thành một mảnh, nối mảnh cuối trước) và bản nhẹ
  `proxy-rev`; `engine.clip_file` dùng bản đó (chưa có thì clip không hiện, `Scene.missing`: `Looks` làm lại khi
  có), xuất video thì làm trước nếu thiếu (`Jobs._reversed_copies`). Video đục cạnh lẻ: H.264 4:4:4.
- **Vùng đứng yên giống hệt từng điểm ảnh** của ảnh gốc / khung đã chọn (kiểm trong selftest và tests).
- **Video ra**: H.264 4:2:0, BT.709, đúng kích thước và số khung/giây của video vào, không âm thanh, **không còn gì
  nêu tên công cụ** (mã hoá vào file tạm rồi remux sạch thẻ, bỏ SEI của x264). Kết xuất nhiều đoạn song song rồi nối
  bằng stream copy. Tiến trình con không tự tìm file trong thư viện: `found_files()` tìm trước.
- Hiệu ứng nhẹ, tự nhiên. Không bẻ cong cả bức ảnh.
- File tải lên là không tin cậy: chỉ đọc bằng mã của mình, chỉ ghi vào thư mục dữ liệu, giới hạn kích thước.
- Mã và chú thích trong mã bằng tiếng Anh; chữ trên trang bằng tiếng Việt, câu dễ hiểu cho người không rành kỹ thuật.

## Kiểm tra và phát hành

- Nhanh: `python -m unittest discover -s tests -t .`. Trọn vẹn: `python bokin_fx.py --selftest`. Giao diện:
  `python packaging/ci_page_test.py` (cần `pip install playwright`; trên Linux đặt `BOKIN_BROWSER` hoặc dùng Chromium
  của Playwright).
- Nhánh mặc định và nhánh phát hành: `claude/relaxed-ptolemy-v732bs`. Push lên đó thì workflow
  `.github/workflows/bokin-fx.yml` chạy trên Windows:
  1. Kiểm thử, thử giao diện trên Edge (ảnh chụp từng bước lưu thành artifact `page-test`), rồi build PyInstaller.
  2. Bản đã build chạy `--selftest`, rồi build bộ cài Inno Setup.
  3. Cài thật, cài đè lúc app đang chạy, rồi thử tự cập nhật lên bản `N.1` với một GitHub giả.
  4. Phát hành Release `fx-v1.0.<số lần chạy>` (ghi SHA-256 vào ghi chú).
- Báo xong chỉ khi CI xanh. Báo đúng kết quả, kể cả khi lỗi.

## Làm việc với chủ dự án

- Trả lời tiếng Việt, câu ngắn, từng bước bấm cụ thể. Chủ dùng Windows, không phải lập trình viên.
- Không xin mật khẩu, khoá, token qua chat. Nhắc che chúng khi chụp màn hình.
- Không tạo pull request nếu chủ không yêu cầu.

## Việc có thể làm sau

- Tự nhận diện vùng: bấm vào vật để khoanh (SAM 2 hoặc MobileSAM, giấy phép Apache-2.0, chạy trên CPU vài giây).
- Kết xuất nhanh hơn nữa (hiện khoảng 5 khung/giây cho 1080p có 7 clip trên máy 4 nhân).
- Keyframe cho thông số clip (độ đậm, vị trí thay đổi theo thời gian).
