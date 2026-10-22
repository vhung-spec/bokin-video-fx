# Bokin FX: ghi chú cho mỗi phiên làm việc

App Windows cho chủ dự án tự dùng trên máy của mình. Thêm hiệu ứng vào video loop rồi xuất video 30 giây hoặc 1 phút,
vẫn loop liền mạch. Giao diện giống Premiere: thư viện hiệu ứng (file ngoài là chính, hiệu ứng tạo sẵn là phụ),
timeline có track và clip, xem nhanh ngay trong trình duyệt, vùng đứng yên, vùng riêng cho từng clip, tiền cảnh che
hiệu ứng. Cách dùng: `README.md`.

## Repo này riêng tư (từ 2026-10-18)

- Kho này công khai tới 2026-10-18; chủ chuyển sang riêng tư để chứa thư viện bản quyền. Vẫn không commit khoá,
  mật khẩu, token, khoá ký hay mã cài đặt của Bokin License.
- `bokin_license/` là thư viện bản quyền chung của các app Bokin, chép nguyên từ kho Bokin License (`VENDORED.txt`:
  phiên bản, commit). Không sửa bản chép ở đây (sửa ở kho đó rồi chép lại). Chỉ phần kiểm tra key và khoá công khai;
  khoá ký và mã cài đặt chỉ nằm trên máy của chủ dự án, không bao giờ vào kho này, bộ cài hay chat.
- Mã hay tài liệu của các kho khác của công ty: chỉ chép vào đây khi chủ đồng ý.
- Bản cài phát hành ở kho **công khai** `vhung-spec/bokin-fx-releases` (chỉ có Releases, không có mã; chủ chọn
  2026-10-19): app đọc Releases ở đó không cần đăng nhập (`fx/updater.py`), khách tải ở
  `…/bokin-fx-releases/releases/latest`. CI đăng bằng secret `RELEASES_TOKEN` (token fine-grained, Contents read/write
  chỉ kho đó, hạn 1 năm: hết hạn thì bước phát hành đỏ, chủ tạo token mới). App 1.0.48 trở về trước đọc Releases của
  kho này (riêng tư, không thấy): cài tay bản mới một lần.
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
- Bản quyền: `fx/licensing.py` (sản phẩm `fx` của `bokin_license`: key Ed25519 gắn Mã máy, có thể có hạn, lưu chung
  `%APPDATA%\BokinLicense\keys.txt`; đọc lại mỗi phút, xuất video thì đọc ngay). Chưa kích hoạt: máy chủ chỉ trả trang,
  `/api/ping`, `/api/license`, `/api/state` và các lệnh của bộ cài / tự cập nhật (`FREE_GET`, `FREE_POST`), còn lại 403
  `{"license": true}`; trang chỉ hiện màn hình kích hoạt (`#lic`, `licenseScreen`; 403 đó giữa chừng: `licenseLost`),
  chân trang `#licChip` (gia hạn). Không thêm cờ hay biến môi trường bỏ qua kiểm tra. Bài thử, selftest, CI giao diện
  tự làm "key center" riêng (khoá ký tạo lúc chạy, thay khoá công khai trong chính tiến trình đó, thư mục key riêng:
  `tests/helpers.py`), không bao giờ dùng khoá thật.
- `fx/index.html`: toàn bộ giao diện (JavaScript thuần, không framework, không bước build), tiếng Việt. Bốn vùng như
  Premiere: Bảng chỉnh hiệu ứng, Chương trình (xem nhanh), Thư viện/Tạo sẵn/Dự án, Timeline. Giao diện tối, nhấn xanh:
  màu, bo góc, khoảng cách, cỡ chữ là token trên `:root` (`--bg`, `--panel`, `--accent`…; màu clip `--k-<loại>` theo
  `data-kind`; thước timeline vẽ canvas đọc `--ruler-*`), biểu tượng nét mảnh tự vẽ trong `ICONS` (`icon()`, nút có
  `data-icon`), không font hay ảnh ngoài.
- Phím tắt: `fx/keys.py` (`ACTIONS`: mã việc, nhóm, nhãn, phím mặc định là `KeyboardEvent.code` sau Ctrl/Alt/Shift,
  `where`: all / edit / paint, hai việc chỉ dùng chung phím khi một cái `edit`, cái kia `paint`). Phím chủ đổi lưu
  trong `keys.json` (chỉ việc đã đổi; phím mặc định trùng phím chủ chọn thì bỏ: `effective`). `/api/keys` (GET, POST),
  bảng **Phím tắt** trên trang (`keysDialog`, `takeKey`); trang hỏi `act(e, id)`, không so phím trực tiếp; nút ghi
  phím bằng `data-tip` (`{id}`), chữ trên trang bằng `keyName(id)`. Thêm việc có phím: thêm vào `ACTIONS`, xử lý trong
  `keydown` của trang, rồi `python -m fx.keys` ghi lại bảng phím mặc định trong `HUONG-DAN.md` (test kiểm).
  `HUONG-DAN.md` (hướng dẫn thao tác cho chủ, đi kèm bản cài, `/api/guide`) hiện ở thẻ Hướng dẫn (`mdHtml`: tiêu đề,
  danh sách, bảng, **đậm**, `phím`). Sửa thao tác trên trang thì sửa cả file đó.
- `fx/timeline.py`: luật thời gian của clip (độ dài 30/60 giây, `Span`: bắt đầu, kết thúc, mờ dần). `index.html` có
  bản sao bằng JavaScript (`span`, `weight`, `plan`, `fileTime`): sửa một bên thì sửa cả bên kia.
- `fx/projects.py`: dự án trong `%APPDATA%\Bokin\FX\projects\<id>` (project.json, input.*, proxy.mp4, still.png,
  masks/, clips/). `layers` là các clip; `tracks` từ dưới lên; `upgrade()` chuyển dự án của bản cũ (một lần);
  `replace_video()` đổi video V1 (file mới tên `input-<x>`, bản nhẹ `proxy-<x>`; thời gian clip giữ theo giây).
  Mọi file của dự án ghi ra file mới rồi đổi chỗ (không sửa tại chỗ), nên `copy()` ("Lưu thành bản khác",
  `/api/project/copy`) dùng hard link được (không được thì chép). Trang tự lưu (`scheduleSave`, mặt nạ); nút **Lưu** /
  Ctrl+S (`saveProject`: `readyToRender`) và dòng `#saveState` ("Đang lưu…", "Đã lưu lúc …", "Chưa lưu được").
- `fx/library.py`: thư viện hiệu ứng dùng chung trong `%APPDATA%\Bokin\FX\library\<id>` (file gốc, ảnh nhỏ, bản nhẹ).
- `fx/engine.py`: ghép một khung (video → vùng đứng yên → các clip theo track, mờ dần), kết xuất song song nhiều đoạn,
  khung chính xác cho trang (`Looks`), clip xem nhanh của hiệu ứng tạo sẵn (`fx_clip`), bản nhẹ của video gốc.
- `fx/effects.py`: phần chung của hiệu ứng tạo sẵn (`Canvas`, `Effect`, `Falling`), hạt sáng, và danh sách `KINDS`
  / `SCHEMA` (nguồn duy nhất của các thông số; trang đọc từ `/api/schema`; `group` = tiêu đề nhóm trên trang, `look` =
  kiểu có sẵn đặt nhiều thông số một lúc; `note` = hướng dẫn trên bảng chỉnh). `fx/snow.py`, `fx/rain.py`,
  `fx/leaves.py`, `fx/glass.py`, `fx/steam.py`, `fx/candle.py`, `fx/frost.py`, `fx/lights.py`: tuyết, mưa, lá
  rơi, nước trên kính, khói, lửa nến, tuyết bám kính, dây đèn, mỗi cái một file. Khói (`fx/steam.py`, mã loại vẫn là `steam`): "Kiểu khói" (`STYLES`: cốc cà phê / trà, nhang, nồi;
  hằng số riêng mỗi kiểu, thanh trượt dùng chung) và "Dáng khói" (`LOOKS`: chỉ đặt chuyển động, lượng khói). Vệt khói
  (streakline) theo luồng khí: điểm toả khói trên cả hình bầu dục, đi lang thang (không thành răng lược); luồng khí
  gom về vài chỗ bốc mạnh đang trôi; sóng dài / vừa / ngắn; xoáy (`_whirls`: cột điểm trôi theo khí, sinh số nguyên
  lần mỗi vòng; làm từ đầu thì làm xoáy trước một vòng đời, `_state`: một khung lẻ tốn bằng 10–15 khung liền nhau);
  nhang có đoạn lặng (`calm`); nồi phả từng đợt, lúc đầu toả mỏng đều trên mặt nước (`vb`, `vt`: không thành hàng
  răng), lượng hơi theo độ rộng mặt nồi (`dense`; hẹp như vòi ấm: luồng phụt nhanh, cách miệng vòi một đoạn mới hiện,
  `jet`, `gap`). Mờ theo căn tuổi, bay hơi khi lên cao và trước hai bên khung (`edge`, mỗi vệt một chỗ); dày quá thì
  phủ chậm dần (`KNEE`, tối đa 95% độ đậm: không thành mảng xám phẳng). Mảnh non nhất của vệt mọc ra từ chỗ toả giữa
  hai tick, độ khói đi dần từ đầu này sang đầu kia mỗi mảnh: không nháy theo tick. Vẽ trên lưới cỡ khung (`GRID_MAX`,
  nhang 1080), các lớp mờ trên lưới thô dần, chỉ phần có khói (`_splat`). Sáng theo cảnh (`LIT`, `light_at()`: 7
  số, đèn chiếu vào (màu theo nguồn sáng, không theo vật sáng như nồi inox), ánh sáng phía sau, phía đèn; hiệu ứng
  đặt một chỗ sáng theo chỗ khung đứng thẳng: `effects.placed_light`, cả tấm xoay và clip xem nhanh; tên clip xem
  nhanh có ánh sáng đó làm tròn (`rounded_light`) khi `lit` > 0: nhích khung không làm lại clip; `Jobs.fx_clip` bỏ
  cả clip đang làm mà không clip nào cần nữa). Hình bầu dục `mark` nằm trong khung ở đáy (`ratio`, `sits`); dự án
  cũ có tâm hình ở đáy khung: `upgrade_in()` (gọi trong `projects.load`) dời khung để tâm hình mới nằm đúng chỗ tâm
  cũ trên hình (cả khi khung đã xoay 3D: qua `overlay.homography`). Lá rơi: chuyển động, thông số trong `fx/leaves.py`; hình lá trong `fx/leafshapes.py`:
  chín kiểu dáng (`SHAPES`: phong 5 thuỳ, phong Nhật, bạch quả, sồi, bàng, bạch dương, liễu, hình tim, cánh hoa) vẽ bằng
  trường khoảng cách (`Lobe`: thuỳ, răng cưa; đầu tròn là nửa ellipse chỉ khi `round_end`: sồi, bàng, còn thuỳ phong
  luôn nhọn; quạt bạch quả; lá tim là đường viền theo góc `_heart_radius`: lệch, răng cưa quanh, mũi ngắn; gân), lỗ bị
  thuỳ khép kín được lấp (`enclosed()`), mỗi kiểu màu riêng cho đỏ/vàng/xanh (`PAINT`: bạch quả không đỏ, sồi đỏ nâu,
  bàng đỏ thẫm tới cam, nâu, gân sáng), mặt trước, mặt sau và hình ngược sáng (`lit`: trang trộn theo hướng lá,
  `light()`, thanh "Nắng chiếu xuyên lá"; cánh hoa sáng vẫn hồng). Mỗi kiểu là một ô tích (`_toggle`,
  không tích ô nào: phong 5 thuỳ); "Loại lá" (`SCENES`, `match`: trang tự chuyển về "Tự chọn" khi tích tay,
  `followChoices`) tích sẵn vài kiểu và tô màu theo mùa; to nhỏ đúng tỉ lệ giữa các kiểu (`REL`, kiểu to nhất đạt "Lá
  to nhất"). Mỗi lá một cỡ hình vừa đủ (`master` theo từng lá và `SHARE` của kiểu), mỗi mẫu chỉ làm một cỡ trong một
  clip (`sizes`, lớn nhất mà lá cần), số mẫu mỗi kiểu giảm khi tích nhiều kiểu để hình của một clip không quá
  `CLIP_SHARE` của `MAX_BYTES` (`_designs()`: vài clip lá cùng lúc không làm hình lại); lá xanh ngả màu (`turning`) chỉ
  khi đã có lá đỏ/vàng; mỗi lần rơi lại là lá khác (`passes()`, `design()`); lá nghiêng chọn hình theo chiều dài, ép
  bằng bộ lọc trung bình (`place()`, chỉ vẽ trong hộp quanh lá `Sprite.box`: cuống dài của bạch quả không tốn thêm);
  lá xa hơi mềm; chao theo tốc độ rơi; clip xem nhanh nhiều điểm ảnh bằng 960x540 (`PREVIEW`,
  `engine.sharper_size`: video dọc giữ cỡ cũ). `upgrade()` của lớp hiệu ứng (gọi trong
  `clean_params`) chuyển thông số clip của bản cũ ("shape" cũ thành ô tích). `VERSION` của hiệu ứng tăng khi hình vẽ ra đổi
  (clip xem nhanh làm lại). Mưa: vệt đều, hai đầu tròn (màn trập nửa khung); một hệ số xa gần cho cả độ rộng và tốc
  độ (hạt mảnh luôn chậm); mỗi lần rơi lại một làn khác (`lanes()`: k làn, bước m / k); hạt chạm đất là một lớp riêng,
  nhiều hơn số vệt vẽ (`_impacts()`: mỗi chỗ k lần mỗi vòng, mỗi lần một chỗ khác). Mưa sáng theo cảnh
  (`Effect.LIT`): `light=` là bản đồ ánh sáng của video (`effects.light_map`, 6 kênh: ánh sáng quanh mỗi chỗ
  (`effects.light()`: mờ rộng + lấp lánh quanh đèn; đèn = chỗ cháy sáng, nhỏ, trong cảnh tối; chỗ cháy sáng thì ít
  nhất là trắng) và ánh sáng tại chỗ; trung bình của khung rải đều cả video; đọc một lần mỗi video:
  `engine.scene_light`, tiến trình con nhận sẵn `_light` qua `found_files` / `fx_clip`; tấm xoay: `sheet.Sheet._lit`).
  Hạt = ánh sáng quanh nó (trước nền tối hơn thì thêm), ít màu riêng; hơi nước = ánh sáng quanh, gần như không màu
  riêng; mỗi kênh cắt ở trắng như máy quay (không làm tối đèn, mặt trời), hạt sáng hơn trắng thì thêm độ phủ
  (`_colours`, bản đồ uint8 làm từ cỡ bản đồ ánh sáng); vẫn là hiệu ứng vẽ đè, nên clip xem nhanh mang đủ; tên clip
  đổi theo video khi `lit` > 0. Mưa sáng theo cảnh đã dời / xoay: clip xem nhanh mang ánh sáng chỗ cũ, nên trang coi
  là gần đúng (`litMoved`, `liveIncomplete`: dừng hình thì hiện khung chính xác). Clip xem nhanh từng loại:
  `PREVIEW` (số điểm ảnh như clip 16:9 rộng chừng ấy, `engine.sharper_size`), `PREVIEW_CRF`
  (mưa: 1280, CRF 42, mưa phùn 34; 30 giây khoảng 20 giây làm, 40 MB; khói nhang CRF 28). Hiệu ứng `lens` (nước trên kính, lửa nến)
  đổi cả hình bên dưới: `apply()` cho video xuất và khung chính xác; trang chỉ phát clip xem nhanh, nên xem nhanh gần đúng và dừng hình thì hiện khung chính xác
  (`liveIncomplete`).
- Hiệu ứng đặt một chỗ (`BOX`: khói, nến): thông số `x`, `y`, `scale` (`_place()`), `box()` cùng công thức với clip
  file (vừa khung). `UNDER` (nến): clip xem nhanh làm từ chính hình dưới hộp (video + vùng đứng yên, `apply()`),
  chỉ giữ phần đổi (`change_rgba`: độ đục vừa đủ), trang đặt nó bằng Normal ngay trên video và vùng đứng yên, dưới
  mọi track (`underBox`), nên clip ở track dưới vẫn hiện (gần đúng). Tên clip đổi theo chỗ đặt, video, và phần vùng
  đứng yên mà mép mềm chạm hộp (`_under_key`, hash điểm ảnh mặt nạ quanh hộp). Clip làm cho chỗ khác thì trang không
  hiện (`elsewhere`, server trả `at`); vẽ vùng đứng yên xong thì trang hỏi lại (`requestAllFx`). `mark` (chỗ khói bốc lên),
  `anchor` (đầu bấc) và `tip` (đỉnh lửa) là vạch vàng trên khung chỉnh clip; đổi cỡ hiệu ứng có `anchor` giữ yên điểm
  đó (`keepAnchor`).
- Lá cây lay động (`fx/foliage.py`): lens chỉ trong vùng của clip (`AREA`; không vùng thì không làm gì, không bao giờ cả
  hình), không đặt chỗ, không xoay. Nắn hình (bilinear) bằng trường dời mượt: vài tầng value noise (lưới cỡ chùm lá,
  theo điểm ảnh khung 1080), mỗi tần số số nguyên vòng mỗi loop; gió từng cơn một hướng (trung bình 0), lá lấp lánh;
  lá động khắp vùng đã tô, chỉ một viền mỏng đứng yên (`edge`: mặc định 2 px của khung 1080, đủ mạnh từ 10 px vào;
  vùng lấy như đã tô, "Mép mềm" của clip không dùng: `engine.area_feather`, trang cũng vậy); chỉ làm trong dải hàng
  có vùng (`bands`). Thông số khớp video mẫu
  (bụi cây 1080p: lệch 0,5–1,5 px, chùm cách trên 30 px gần như độc lập, chu kỳ 2–6 giây). `UNDER` không hộp: clip xem
  nhanh là video + vùng đứng yên cỡ 1280 (`PREVIEW`), hiệu ứng trong vùng của clip, chỉ giữ phần đổi (`change_rgba`
  theo dải `_bands_of`, WebM có thẻ BT.709); tên theo hash vùng của clip (`_area_painted`), vùng đứng yên gần đó;
  trang hỏi lại khi tô vùng của clip, đặt nó trong `underBox`; clip làm cho đúng vùng hiện tại thì xem nhanh là
  chính xác (`areaLive`: dừng hình không cần khung chính xác).
- Nhấp nháy vùng (`fx/flicker.py`, mã loại `flicker`): lens trong vùng của clip như lá cây lay động (`AREA`, `UNDER`
  không hộp, không xoay; "Mép mềm" của chính nó, mềm vào trong mép tô). Nhân hình với mức sáng theo thời gian (`level()`:
  tổng sóng số nguyên vòng mỗi loop; nến: lung linh + thỉnh thoảng chao, lửa, nhấp nháy đều, thở, đèn chập chờn), chỗ
  sáng theo nhiều hơn (`focus`), sáng lên thì trắng hơn, tối đi thì đỏ hơn (`WARM`: xanh dương theo nhiều nhất), lửa
  lắc làm một bên sáng hơn (`sway`). Lời nhắc khi tô vùng và khi chưa tô lấy từ schema (`area_tip`, `area_empty`).
- Tuyết bám kính (`fx/frost.py`, mã loại `frost`): hiệu ứng vẽ đè (không lens: xem nhanh là chính xác), phủ cả hình,
  xoay được. Hoa băng làm một lần theo chiều cao khung (`_segments`: thân lượn, gai 50° hình lông chim, cành phụ, rẽ
  nhánh `_stem`; nửa số mọc từ mép, nửa từ trong lớp băng, hướng bất kỳ; "Cỡ hoa băng" `fern_k`: nhỏ thì nhiều hơn,
  gai dày hơn, nét mảnh hơn), vẽ bằng điểm rải dọc nét lên điểm ảnh (`_ink`: mực mỗi điểm ảnh, độ "mọc" sớm nhất `first`);
  sương muối, kính mờ trên dải viền (`_field`: khoảng cách tới mép, góc dày theo `corner`, viền trong lượn bằng
  noise); "Băng lan ra, thu vào" (`creep`): một vòng mỗi loop, chỉ hiện phần đã mọc. Tuyết đọng mép dưới (`_sill`,
  tĩnh), bông tuyết đậu rồi tan (mỗi slot k lần mỗi vòng, mỗi lần một chỗ: `places`; sao băng, đĩa, cụm vụn), lấp
  lánh trên băng. Sáng theo cảnh (`LIT` như mưa, thanh `lit`): sương muối, kính mờ là ánh sáng tại chỗ của bản đồ ánh
  sáng làm mờ (`_behind`: kính mờ thật, hơi xám đi) + chút màu riêng (ánh đèn trong phòng, `*_ROOM`); hoa băng sáng
  hơn, nổi khối theo hướng sáng trên trái (`_colours`); tuyết, bông tuyết cũng vậy. Clip xem nhanh cỡ 1280
  (`PREVIEW`).
- Dây đèn lấp lánh (`fx/lights.py`, mã loại `lights`): hiệu ứng đặt một chỗ (`BOX` vuông), ánh sáng (Screen: clip xem
  nhanh H.264 nền đen), xoay được. Dây giăng một lần trong hộp: quấn quanh hình nón (xoắn ốc nhìn từ ngang giữa cây:
  `EYE`, `AWAY`; bóng phía sau cây mờ theo `back`), võng, rèm, rải đều; bóng cách đều theo chiều dài dây 3D
  (`_even`), lệch nhẹ (`jitter`), màu lần lượt dọc dây (`PALETTES`). Mỗi bóng: lõi gần trắng, quầng màu, tia sáng khi
  loé; "Chỉ ánh sáng" (`bulb` = `glow`): không lõi, chỉ quầng mềm + tia. Kiểu nháy (`levels()`): từng bóng, nhóm thở
  lần lượt, chạy đuổi dọc dây (`CHASE` bóng một sóng), hai nhóm, sáng đứng; mọi tần số số nguyên vòng mỗi loop.
  "Theo đèn có sẵn trong video" (`layout` = `scene`): `fx/bulbs.py` tìm chấm sáng nhỏ (cực đại cục bộ, sáng hơn xung
  quanh, đỉnh phẳng chỉ một điểm, cách nhau vài px) trong vài khung của video, trong hộp và vùng của clip; lưu trong
  thông số `spots` (kiểu `points` của schema: `effects._points`, trang không vẽ thanh mà vẽ nút **Tìm đèn trong video**,
  `pointsRow`, `findBulbs`, `/api/bulbs`; dời / đổi cỡ hộp thì trang tự tìm lại, `refindBulbs`) theo chỗ trong hộp, độ
  sáng, màu (màu quầng quanh bóng).
- `fx/sprites.py`: hình hạt vẽ bằng mã (chấm, quầng sáng). `fx/overlay.py`: clip từ video/ảnh ngoài (`timeline()`
  cho clip suốt video, `playing()` cho clip ngắn hơn, xoay 3D: `homography()`, `quad()`).
- `fx/sheet.py`: hiệu ứng tạo sẵn dời / đổi cỡ / xoay 3D như clip file (`Effect.TURNS`, nến không xoay). Vẽ trên một
  tấm (`effects.sheet_rect`: hiệu ứng phủ cả hình là tấm hình khung, đặt như file cỡ khung "contain"; hiệu ứng đặt một
  chỗ: hộp của nó) ở cỡ phép nắn cần (không mịn hơn cỡ đặt, tối đa `SHEET_MOST` khung), nắn bằng cùng `homography()`
  (bilinear), rồi đặt như clip file (`overlay.Laid`, `overlay.lay`: hoà trộn, mờ dần, vùng và tiền cảnh sau khi nắn).
  Lens (nước trên kính): đưa hình dưới tấm lên tấm, `apply(..., kept)` (lens xoay được phải cho biết giữ lại bao
  nhiêu của mỗi điểm ảnh); phần đổi là một lớp (phần thêm = màu nhân sẵn, 1 - kept = độ phủ), nắn và đặt như hiệu ứng
  vẽ: hình khung * kept + phần thêm. Chỗ sương mờ mất chi tiết như khi thẳng (chỉ cộng phần đổi after - before thì chi
  tiết khung lọt qua sương); chỗ không đổi và ngoài tấm giữ nguyên từng điểm ảnh. Các tấm của một cảnh dùng chung bộ
  nhớ (`Scratch`: một canvas, cả của hiệu ứng thẳng, một bộ đệm lens, cỡ lớn nhất). Mỗi clip xoay tốn thêm một lần
  nắn (1080p: 1,5–2 lần clip thẳng; tấm to hơn khung, ví dụ 140% để nghiêng vẫn phủ kín: 2–2,5 lần; clip xoay chồng
  nhau thì cộng dồn); tấm quá `SHEET_MOST` thì vẽ thưa hơn (cỡ trên khoảng 150%: hơi mềm, đổi lấy bộ nhớ). 100%, ở
  giữa, không xoay: đường cũ, giống hệt từng byte (`effects.on_sheet`). Chỗ và góc không
  vào clip xem nhanh (`effects.sheet_params`, `SHEET_KEYS`): xoay là thấy ngay, tên clip không đổi. Hiệu ứng trên tấm
  lấy đúng tỉ lệ của video (`Effect(..., aspect)`): số hạt (lá) theo tỉ lệ, lệch một hạt là cả loạt lá dời chỗ.
- `fx/video.py`: ffmpeg (bản đi kèm imageio-ffmpeg): đọc khung, ghi H.264, nối các đoạn bằng stream copy, bản nhẹ cho
  trình duyệt (H.264, hoặc VP9 WebM khi có nền trong suốt).
- `fx/jobs.py`: hai hàng đợi: xuất video, và chuẩn bị xem nhanh (bản nhẹ, clip hiệu ứng tạo sẵn).
- `fx/updater.py`: tự cập nhật. `fx/selftest.py`: tự kiểm tra.
- `packaging/`: PyInstaller (`BokinFX.spec`), Inno Setup (`BokinFX.iss`), icon, bài thử cài đặt, cập nhật, và giao
  diện trên Edge (`ci_page_test.py`, dùng Playwright) trong CI.
- `brand/`: bộ logo chung của mọi app Bokin (hướng B, chủ chọn 2026-10-19; `brand/README.md`): ô màu riêng mỗi app, mã
  hai chữ màu đậm góc dưới trái, vòng lặp nhỏ góc trên phải (bỏ dưới 32 điểm ảnh). `make_icons.py` chỉ dùng Pillow và
  numpy: chữ là đường viền Be Vietnam Pro ExtraBold (OFL) trong `glyphs.json`, tô chẵn lẻ; `write_ico()` vẽ riêng
  từng cỡ; `svg()`, `small_svg()`; `out/` là file cho mọi app (commit sẵn). Bokin FX: `packaging/make_icon.py` (CI
  chạy trước PyInstaller, `icon.ico` không commit), tab và logo góc trên của trang là `small_svg("fx")`. Đổi logo thì
  chạy lại `python brand/make_icons.py` và dán lại đoạn SVG vào `index.html`.
- Thư viện: numpy, Pillow, imageio-ffmpeg. **Không dùng OpenCV**: OpenCV cần Media Foundation, mà Windows bản N và
  Windows Server không có sẵn.

## Xem nhanh trên trang

- Trình duyệt tự ghép: `<video>` bản nhẹ của video gốc, ảnh vùng đứng yên, rồi từng clip với CSS `mix-blend-mode`
  (screen, multiply, plus-lighter cho add), `opacity` (mờ dần) và `mask-image` (vùng đã làm mềm; clip "sau tiền cảnh"
  thêm tiền cảnh đảo ngược, `mask-composite: intersect`). Một đồng hồ chung giữ mọi video khớp nhau.
- Hiệu ứng tạo sẵn: app kết xuất cả độ dài video thành clip nhỏ (rộng 640, hoặc `PREVIEW` của hiệu ứng: lá rơi 960;
  WebM trong suốt, hoặc H.264 cho hạt sáng dùng Screen). Tên file theo nội dung (`fx-<khoá>.webm`), hai nửa của clip
  bị cắt dùng chung.
- Khung chính xác từ `engine.Looks` (`/api/look`) là thứ video xuất ra. Trang chỉ hiện nó khi dừng hình nếu người dùng
  bật nút Khung chính xác (`#tExact`), hoặc khi bản xem nhanh còn thiếu (`liveIncomplete`). Không tự thay ảnh khi
  dừng: mắt thấy như hình bị giật.
- Khi phát, video gốc là đồng hồ; dừng thì giữ khung đang hiện (`requestVideoFrameCallback`). Bản nhẹ đánh số khung
  đều (`setpts=N/(fps*TB)`, `-fps_mode cfr`), nên khung n của trang là khung n của app.
- Vẽ vùng: nét cọ vẽ thẳng lên mặt nạ và lớp màu; việc nặng (gửi vùng, làm mềm cho xem nhanh) chỉ chạy khi ngừng vẽ.
- Khung chỉnh clip (`#xform`, `placed()`): cùng công thức đặt clip với `overlay.Overlay` (và `Effect.box()` của
  hiệu ứng tạo sẵn đặt một chỗ: `BOX`, `placeable()`; clip xem nhanh của nó chỉ là cái hộp, trang đặt vào chỗ).
  Vạch vàng `#xformMark` theo `mark` / `anchor` của schema.
- Xoay 3D của clip file (`rot_x`, `rot_y`, `rot_z`, `persp`): `overlay.homography()` / `quad()` và trang (`homography`,
  `quad`, `turnCSS`) dùng cùng một ma trận, đúng như CSS `perspective(d) rotateX rotateY rotateZ scale(lật)` quanh tâm
  clip (y hướng xuống; `d` theo đường chéo khung, lùi ra khi một góc tới quá gần mắt: `NEAR`). Renderer nắn hình bằng
  Pillow `PERSPECTIVE` (bicubic), chỉ đọc phần nguồn đang hiện trong khung, mép mềm `coverage()`. Ba góc bằng 0: đường
  cũ, giống hệt từng byte. Phần nguồn đọc ở cỡ phép nắn cần (`jacobians()`: một điểm ảnh của phần đọc không lớn hơn
  một điểm ảnh khung chỗ hình to nhất, không mịn hơn file gốc), mép lặp ra đủ xa cho phía xa của mặt nghiêng
  (`pad_for()`). `#xform` xoay theo clip (bỏ lật), nút tròn `#xformRot` xoay phẳng (giữ con trỏ riêng: bấm đúp là
  của nó), nút `#xform3d` nằm cạnh nút tròn (không ở giữa: giữa khung là để dời) và nghiêng; nhãn tên và số
  (`#xformLab`) chỉ hiện lúc đang kéo, ở góc trên trái (lúc yên nó che đúng chỗ đang căn, như cái cốc dưới khói). Chọn clip trên hình theo tứ giác (`hitsClip`, `pickAt`); hiệu ứng phủ kín khung chỉ được chọn ở chỗ không có
  clip khác (`coversFrame`). Đang chọn nó: kéo trong khung là dời nó; bấm (không kéo) trúng clip nhỏ hơn thì lúc nhả
  chuột chọn clip đó (`handoff`; clip phủ kín khung không giành, `fillsFrame`). Hiệu ứng tạo sẵn trừ nến xoay như vậy
  (`turns()`, `sheetKind()`: clip xem nhanh cả khung hoặc hộp, đặt bằng cùng `turnCSS`). CI so góc trang với Python
  (clip file, tuyết, khói, hạt sáng và mưa trong Mix).
- Màu của clip file: `overlay.colour_ops` (đường cong tông giữ đen là đen, rồi ma trận màu) và trang dựng đúng các số
  đó thành bộ lọc SVG (`colourOps`, `colourFilter`, `feComponentTransfer` bảng + `feColorMatrix`, sRGB). "Khớp màu
  với cảnh" (`fx/colourmatch.py`, `/api/match`, trang `matchColours` sau `saveNow`): đo video ở chỗ clip nằm (hộp quanh
  tứ giác, rộng thêm `MARGIN`, trong vùng riêng nếu có; vài khung trong thời gian clip) và phần hiện ra của file (Normal:
  điểm đục; Screen/Add: điểm sáng; Multiply: điểm tối), rồi đặt chính các thanh màu, khớp trên điểm ảnh của clip qua
  `colour_ops`: độ sáng, độ tương phản (chỉ giảm), độ rực (lệch khỏi ám màu riêng, không tính ám màu), rồi ám màu (ấm
  + phủ màu của cảnh, tìm lưới), đi `STRENGTH` quãng đường. Screen/Add chỉ tối đi.
- Chép clip (`clipboardOf`, `copyClip`, `pasteClip`, `trackMenu`, `pasteHere`): `S.clip` giữ bản chép cả khi mở dự án
  khác (cùng trang); vùng riêng chép theo chỗ trên khung (co giãn theo cỡ mặt nạ). Dán: đầu phát, track dưới chuột
  (`S.tlMouse`), hoặc track của clip đang chọn (hết chỗ: track mới); chuột phải chỗ trống: dán đúng chỗ đó. Clip suốt
  video dán vẫn suốt video; mờ dần 0,5 giây nếu clip gốc nối liền (`joinFades` bỏ lại khi nối tiếp).
  "Dán thiết lập" (Paste Attributes, `Ctrl+Alt+V`, `pasteAttrsDialog`, `pasteAttrs`): thông số của `S.clip` theo nhóm
  như bảng chỉnh (`paramGroups`: clip file `FILE_GROUPS`, hiệu ứng tạo sẵn theo `group` của schema) và của clip
  (`CLIP_GROUPS`: mờ dần; mép mềm vùng + sau tiền cảnh; không chép hình vùng). Khác loại: chỉ `SHARED_GROUPS` (vị trí và
  cỡ, xoay 3D) và nhóm của clip. Giá trị kiểm theo schema của clip nhận (`fitValue`). Vào clip đang chọn hoặc mọi clip
  cùng loại; một bước hoàn tác.
- Hoàn tác: `S.hist` giữ timeline và cả lần vẽ vùng (mặt nạ trước/sau, chỉ 8 lần gần nhất). Phóng to hình:
  `zoomPicture` (Ctrl +/-/0/1, `+` `-` khi chuột trên hình); Ctrl + lăn chuột không phóng cả trang.
- Dời hình như Premiere (`layoutScreen`, `panBy`, `unfit`): `S.zoom` là `"fit"` (giữa, vừa khung) hoặc số điểm ảnh
  video trên một điểm màn hình, nhỏ tới 10% (`minScale`); `S.panX/Y` dời ở mọi mức (bàn tay `H`/`#handBtn`, chuột
  phải/giữa, phím cách, hai ngón trên bàn di chuột, Shift + lăn: `wheelPans`), luôn còn `PAN_KEEP` điểm của khung
  trên màn hình. Phím cách phát/dừng lúc nhả, trừ khi đã dời hình lúc giữ (`S.spaceKey`); giữ `H` mà dời thì nhả là
  thôi bàn tay (`S.handKey`). Phím phóng (`zoomKey`) phóng quanh chuột khi chuột trên hình. Quanh khung là vùng tối
  (`#frame`: viền khung + lớp che); lớp của clip đang chọn tràn ra ngoài khung (`.spill`, `spill()`) nên thấy mờ ở
  đó, `#xform` kéo được ở đó; ngoài khung chỉ bấm trúng clip đang chọn. Dời hình không hỏi lại khung chính xác.
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
  4. Phát hành Release `fx-v1.0.<số lần chạy>` ở `vhung-spec/bokin-fx-releases` (ghi SHA-256 vào ghi chú).
- Báo xong chỉ khi CI xanh. Báo đúng kết quả, kể cả khi lỗi.

## Làm việc với chủ dự án

- Trả lời tiếng Việt, câu ngắn, từng bước bấm cụ thể. Chủ dùng Windows, không phải lập trình viên.
- Không xin mật khẩu, khoá, token qua chat. Nhắc che chúng khi chụp màn hình.
- Không tạo pull request nếu chủ không yêu cầu.

## Việc có thể làm sau

- Tự nhận diện vùng: bấm vào vật để khoanh (SAM 2 hoặc MobileSAM, giấy phép Apache-2.0, chạy trên CPU vài giây).
- Kết xuất nhanh hơn nữa (hiện khoảng 5 khung/giây cho 1080p có 7 clip trên máy 4 nhân).
- Keyframe cho thông số clip (độ đậm, vị trí thay đổi theo thời gian).
