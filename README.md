# Facility Management Vision Analytics — Smart Parking System

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-DNN%20ONNX-5C3EE8.svg)](https://opencv.org/)
[![Database](https://img.shields.io/badge/SQLite-WAL%20Mode-003B57.svg)](https://www.sqlite.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Triple--Check%20Verification-success.svg)]()

Sistem analitik CCTV cerdas berbasis *Computer Vision* untuk pemantauan ketersediaan petak parkir secara real-time dengan arsitektur bebas margin error (**Zero-Margin Error Architecture**). Dirancang khusus untuk lingkungan operasional fasilitas komersial dan koridor industri dengan efisiensi komputasi tinggi (CPU-friendly via OpenCV DNN).

---

## Daftar Isi
1. [Ringkasan Sistem](#1-ringkasan-sistem)
2. [Logika Utama: Triple-Check Parking Verification](#2-logika-utama-triple-check-parking-verification)
   - [Arsitektur Validasi 3 Pilar](#arsitektur-validasi-3-pilar)
   - [Finite State Machine Petak Parkir](#finite-state-machine-petak-parkir)
   - [Anti-ID Churn & Spatial Slot Anchoring](#anti-id-churn--spatial-slot-anchoring)
   - [Initial Cold-Start Scan & Baseline Occupancy](#initial-cold-start-scan--baseline-occupancy)
3. [Spesifikasi Endpoint API & Streaming](#3-spesifikasi-endpoint-api--streaming)
   - [Pemisahan Peran: Telemetri vs Visual Stream](#pemisahan-peran-telemetri-vs-visual-stream)
   - [Daftar Endpoint Utama](#daftar-endpoint-utama)
4. [Struktur Direktori Repositori](#4-struktur-direktori-repositori)
5. [Instalasi & Panduan Pengoperasian](#5-instalasi--panduan-pengoperasian)
   - [Persiapan Lingkungan Virtual (.venv)](#1-persiapan-lingkungan-virtual-venv)
   - [Konfigurasi Lingkungan (.env)](#2-konfigurasi-lingkungan-env)
   - [Menjalankan Sistem Utama](#3-menjalankan-sistem-utama)
   - [Kalibrasi ROI Spasial (GUI Calibrator)](#4-kalibrasi-roi-spasial-gui-calibrator)
   - [Verifikasi Pengujian & Regression Tests](#5-verifikasi-pengujian--regression-tests)

---

## 1. Ringkasan Sistem

Sistem ini memantau area parkir aktif (`cam_01`) menggunakan kamera IP melalui protokol RTSP. Berbeda dengan pendekatan deteksi objek konvensional yang rentan terhadap kedipan deteksi (*detection flicker*), oklusi, dan pertukaran ID tracker (*ID churn*), sistem ini mengimplementasikan **Zero-Margin Error Architecture** yang mengunci status okupansi pada koordinat fisik petak lantai secara persisten.

### Keunggulan Utama:
- **Decoupled Resolution Pipeline**: Ingest RTSP native pada resolusi 1080p (`1920x1080`) untuk audit snapshot berdefinisi tinggi, dipadukan dengan kanvas inferensi simetris 16:9 (`640x360`) untuk meminimalkan beban CPU dan memori.
- **CPU-Friendly DNN Inference**: Menjalankan model YOLO11n ONNX langsung melalui `cv2.dnn` tanpa memerlukan dependensi runtime berat (seperti CUDA atau PyTorch).
- **Minimalist Clean Visual**: Kanvas live stream yang bebas dari polusi teks atau bounding box mengambang, hanya menampilkan kontur poligon status, hairline tripwire, dan HUD kapasitas 1 baris ringkas.

---

## 2. Logika Utama: Triple-Check Parking Verification

Untuk mengeliminasi kesalahan hitung (*margin error*), sistem menerapkan verifikasi silang tiga pilar analitik sebelum sebuah petak parkir diputuskan terisi atau kosong:

```
                  ┌────────────────────────────────────────────────────────┐
                  │            Pilar 1: YOLO Vehicle Detection            │
                  │        (Deteksi Objek 'car' + Estimasi Roda)           │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
       ┌─────────────────────────────┐               ┌─────────────────────────────┐
       │   Pilar 2: Polygon Dwell    │               │  Pilar 3: Tripwire Gate     │
       │    (zone_01 s.d. zone_08)   │               │     (tw_01 s.d. tw_08)      │
       │  Point-in-Polygon >= 10.0s  │               │   A->B Masuk / B->A Keluar  │
       └──────────────┬──────────────┘               └──────────────┬──────────────┘
                      │                                             │
                      └──────────────────────┬──────────────────────┘
                                             ▼
                            ┌─────────────────────────────────┐
                            │    Triple-Check State Machine   │
                            │   VACANT ➔ ENTERING ➔ OCCUPIED  │
                            └─────────────────────────────────┘
```

### Arsitektur Validasi 3 Pilar

1. **Pilar 1: Vehicle Detection (YOLO Inference)**
   - Mendeteksi keberadaan fisik kendaraan (`car`, `bus`, `truck`) di setiap frame aktif.
   - Titik evaluasi spasial utama menggunakan estimasi kontak tapak roda bawah (**wheel contact point**):
     $$\text{wheel\_pt} = \left(\frac{x_1 + x_2}{2}, y_2\right)$$
   - Dilengkapi titik evaluasi cadangan: *lower-body center* ($y_1 + 0.75 \times h$) dan *centroid bounding box*.

2. **Pilar 2: Static Polygon Slots (Dwell Occupancy)**
   - Memantau 8 petak fisik (`zone_01` s.d. `zone_08`) yang telah dikalibrasi pada koordinat aspal riil.
   - Menguji apakah titik tumpu kendaraan berada di dalam batas petak menggunakan algoritma `cv2.pointPolygonTest`.
   - Kuota parkir hanya dipotong jika kendaraan stasioner berada di dalam poligon selama waktu inap valid (**dwell time** $\ge 10.0\text{ detik}$).

3. **Pilar 3: Directional Tripwire Gates (Gatekeeper Virtual)**
   - Garis tripwire berpasangan (`tw_01` s.d. `tw_08`) dipasang tepat di bibir pembatas tiap petak parkir.
   - Memvalidasi fase manuver kendaraan melintasi gerbang petak:
     * Arah **A $\rightarrow$ B** (*Masuk Slot*): Memicu transisi awal slot dari `VACANT` menuju `ENTERING`.
     * Arah **B $\rightarrow$ A** (*Keluar Slot*): Memicu transisi slot dari `OCCUPIED` menuju `LEAVING`.
   - Membatalkan manuver palsu (*false entry*) jika kendaraan melintas masuk lalu segera mundur kembali keluar.

---

### Finite State Machine Petak Parkir

Setiap petak parkir dikelola oleh mesin status independen dengan 4 fase:

```
                     A->B Crossing (Masuk)
       ┌─────────────────────────────────────────────────┐
       ▼                                                 │
  ┌──────────┐   Dwell >= 10s   ┌──────────┐  B->A (Keluar)   ┌─────────┐
  │  VACANT  ├─────────────────►│ ENTERING ├─────────────────►│ LEAVING │
  └────▲─────┘   (atau direct)  └────┬─────┘   Clear >= 3.0s  └────┬────┘
       │                             │                             │
       │                     Dwell >= 10.0s                        │
       │                             ▼                             │
       │                        ┌──────────┐                       │
       └────────────────────────┤ OCCUPIED ├───────────────────────┘
          Clear Timeout > 2.0s  └──────────┘  Konfirmasi Fisik Kosong
```

- **`VACANT` (Kosong)**: Petak tidak berpenghuni. Outline dirender berwarna Cyan/Emerald halus tanpa teks di lantai.
- **`ENTERING` (Proses Masuk)**: Sinyal tripwire $A \rightarrow B$ terpicu atau kendaraan mulai memasuki poligon. Sistem menunggu konfirmasi dwell 10 detik.
- **`OCCUPIED` (Terisi Penuh)**: Kendaraan terbukti diam $\ge 10\text{ detik}$. Kuota parkir dipotong. Petak dirender dengan outline merah 1px, arsir transparan lembut ($\alpha = 0.10$), dan mini badge centroid `[S{num}]`.
- **`LEAVING` (Proses Keluar)**: Kendaraan melintasi tripwire arah keluar ($B \rightarrow A$) atau menghilang dari poligon.

---

### Anti-ID Churn & Spatial Slot Anchoring

Di lapangan, inferensi tracker kerap mengalami pergantian ID pelacak (*ID churn / track switching*, misalnya track `#12` berkedip dan berganti menjadi `#230`). Jika penguncian status bergantung pada kesamaan numerik `track_id`, kuota parkir akan berkedip (*flip-flop*).

Sistem ini menerapkan **Reversed Ownership Architecture (Slot-Centric Anchoring)**:
1. **Kepemilikan Berpusat pada Slot**: Poligon petak bertindak sebagai *anchor* utama, bukan `track_id` kendaraan.
2. **Silent ID Update**: Jika track baru muncul dengan ID berbeda di slot yang sama dalam jendela toleransi temporal $\le 3.0\text{ detik}$, sistem menguji:
   - Titik roda track baru berada di dalam poligon slot, ATAU
   - IoU antara bounding box baru dengan bounding box lama $\ge 0.35$.
   Jika salah satu terpenuhi, sistem memperbarui referensi ID secara senyap **tanpa mereset dwell timer atau fase okupansi**.
3. **Hysteresis Konfirmasi Kosong (`polygon_clear_since`)**: Slot `OCCUPIED` tidak akan langsung beralih ke `LEAVING` hanya karena objek hilang selama 1-2 frame. Poligon harus terbukti bersih tanpa kendaraan selama minimal **$3.0\text{ detik}$ berturut-turut** sebelum status dinyatakan lepas.
4. **Re-Park Debounce**: Jika kendaraan yang sedang bermanuver keluar (`LEAVING`) kembali terdeteksi stasioner di dalam petak dalam kurun waktu 5 detik, status otomatis dikembalikan ke `OCCUPIED`.

---

### Initial Cold-Start Scan & Baseline Occupancy

Saat sistem baru pertama kali dinyalakan (atau di-restart), umumnya sudah ada kendaraan yang terparkir statis sejak awal.
- **Bypass Motion Gating**: Selama $30\text{ frame}$ pertama (fase warmup), modul motion gating di-bypass penuh untuk memastikan inferensi YOLO memindai seluruh kanvas.
- **Instant Baseline Occupancy**: Kendaraan yang terdeteksi di dalam poligon petak pada masa warmup langsung dikonfirmasi berstatus `OCCUPIED` dengan dwell awal diset ke batas ambang (10.0s), sehingga status kuota parkir langsung akurat sejak detik pertama booting.

---

## 3. Spesifikasi Endpoint API & Streaming

Sistem menerapkan pemisahan tugas yang tegas (*Separation of Concerns*): antarmuka visual stream dibuat sangat bersih untuk kebutuhan monitor CCTV, sedangkan rincian telemetri lengkap dialirkan melalui REST API.

```
                      ┌───────────────────────────────────────┐
                      │    Facility Vision Hub (Port 8070)    │
                      └───────┬───────────────────────┬───────┘
                              │                       │
           REST API (JSON Data)                       Video MJPEG Stream
                              ▼                       ▼
                   ┌─────────────────────┐ ┌─────────────────────┐
                   │  GET /api/cameras   │ │ GET /video_feed/    │
                   │  GET /api/status/   │ │       cam_01        │
                   └──────────┬──────────┘ └──────────┬──────────┘
                              │                       │
                              ▼                       ▼
                     Videotron / Dashboard       CCTV Monitor Kiosk
```

### Pemisahan Peran: Telemetri vs Visual Stream
- **Video Stream (`/video_feed/cam_01`)**: Aliran MJPEG bersih tanpa teks berlebih. Menampilkan kontur petak parkir (Cyan untuk Kosong, Merah transparan untuk Terisi) serta status kapasitas 1 baris ringkas di pojok kanan atas:
  ```text
  PARKING: 1/7 AVAILABLE | TERISI: [S1, S2, S4, S5, S6, S7]
  ```
- **REST API (`/api/cameras` & `/api/status/cam_01`)**: Menyediakan data terstruktur untuk integrasi IoT, videotron pintu gerbang, dan sistem manajemen gedung.

---

### Daftar Endpoint Utama

#### 1. Kuota Seluruh Kamera: `GET /api/cameras`
Mengembalikan daftar ringkas kapasitas parkir dan metrik gateway:
```json
[
  {
    "camera_id": "cam_01",
    "display_name": "Kamera 01 (Koridor Utama)",
    "status": "active",
    "processed_count": 81693,
    "motion_detected_count": 3510,
    "resolution": "1920x1080 -> 640x360",
    "parking": {
      "total_slots": 7,
      "occupied_slots": 5,
      "available_slots": 2,
      "gate_in": 15,
      "gate_out": 0
    }
  }
]
```

#### 2. Detail Status Per-Petak: `GET /api/status/{camera_id}`
Mengembalikan telemetri mendalam per masing-masing slot (`zone_01` s.d. `zone_07`):
```json
{
  "camera_id": "cam_01",
  "display_name": "Kamera 01 (Koridor Utama)",
  "status": "active",
  "parking": {
    "total_slots": 7,
    "occupied_slots": 5,
    "available_slots": 2,
    "gate_in": 15,
    "gate_out": 0,
    "slots": {
      "zone_01": { "phase": "OCCUPIED", "occupied": true, "track_id": 56, "dwell": 3855.5 },
      "zone_02": { "phase": "OCCUPIED", "occupied": true, "track_id": 56, "dwell": 3348.4 },
      "zone_03": { "phase": "VACANT",   "occupied": false, "track_id": null, "dwell": 0.0 },
      "zone_04": { "phase": "OCCUPIED", "occupied": true, "track_id": 34, "dwell": 3854.9 },
      "zone_05": { "phase": "VACANT",   "occupied": false, "track_id": null, "dwell": 0.0 },
      "zone_06": { "phase": "OCCUPIED", "occupied": true, "track_id": 7,  "dwell": 3855.5 },
      "zone_07": { "phase": "OCCUPIED", "occupied": true, "track_id": 30, "dwell": 3855.5 }
    }
  }
}
```

#### 3. Log Audit Kejadian: `GET /api/incidents`
Mengambil 50 rekaman riwayat insiden dan perpindahan fase kendaraan dari database SQLite WAL.

#### 4. Live Visual Feed: `GET /video_feed/{camera_id}`
Aliran visual berlatar belakang MJPEG kontinu dengan latensi rendah untuk tampilan peramban.

#### 5. Web UI Kiosk: `GET /`
Dasbor antarmuka pengguna web zero-scroll (100vh) responsif dengan mode fokus kamera.

---

## 4. Struktur Direktori Repositori

```text
facility_manajement/
├── cameras/                     # Konfigurasi per-kamera
│   ├── _template/               # Template konfigurasi standar
│   └── cam_01/                  # Konfigurasi kamera aktif (Smart Parking)
│       ├── config.json          # Parameter kamera, RTSP stream, dan resolusi
│       └── roi_zones.json       # Koordinat poligon petak (zone_XX) & tripwire (tw_XX)
├── engine/                      # Inti modul pemrosesan analitik
│   ├── config_loader.py         # Parser konfigurasi & validasi skema
│   ├── detector.py              # Wrapper YOLO ONNX via OpenCV DNN
│   ├── geometry.py              # Perhitungan geometris spasial (poligon & crossing)
│   ├── logger.py                # Sistem pencatatan log terstruktur
│   ├── pipeline.py              # CameraOrchestrator & streaming pipeline loop
│   ├── preprocessor.py          # Decoupled scaling & motion gating adaptif
│   ├── smart_parking.py         # Triple-Check Parking Engine & State Machine
│   └── tracker.py               # Centroid multi-object tracker dengan coasting
├── storage/                     # Manajemen database & bukti rekaman
│   ├── database.py              # SQLite connection pool (WAL mode)
│   └── models.py                # Skema data insiden & model ORM
├── tools/                       # Perangkat pengujian & utilitas kalibrasi
│   ├── audit_schema.py          # Validasi skema konfigurasi sebelum startup
│   ├── roi_calibrator.py        # GUI interaktif untuk penarikan poligon & tripwire
│   ├── test_cold_start_parking.py # Pengujian cold-start baseline scan
│   ├── test_smart_parking.py    # Pengujian logika dasar Smart Parking
│   └── test_triple_check_parking.py # Pengujian komprehensif Triple-Check & Anti-Churn
├── web/                         # Antarmuka web dasbor & API server
│   ├── app.py                   # FastAPI application factory & REST endpoints
│   ├── buffer.py                # In-memory circular buffer O(1) per kamera
│   └── templates/               # Templat antarmuka HTML Jinja2
├── weights/                     # Bobot model neural network
│   └── yolo11n.onnx             # Model YOLO11n ONNX resmi
├── main.py                      # Titik masuk utama aplikasi (Application Launcher)
├── run_system.bat               # Skrip peluncuran otomatis untuk lingkungan Windows
├── requirements.txt             # Daftar dependensi Python
├── Dockerfile                   # Definisi kontainer Docker
└── docker-compose.yml           # Konfigurasi orkestrasi kontainer Docker
```

---

## 5. Instalasi & Panduan Pengoperasian

### 1. Persiapan Lingkungan Virtual (.venv)

Pastikan Python 3.10, 3.11, atau 3.12 telah terpasang pada sistem:

```bash
# 1. Masuk ke direktori proyek
cd /d D:\Project\facility_manajement

# 2. Buat lingkungan virtual python
python -m venv .venv

# 3. Aktifkan lingkungan virtual
# Pada Windows PowerShell:
.venv\Scripts\Activate.ps1
# Pada Windows Command Prompt:
.venv\Scripts\activate.bat

# 4. Pasang seluruh dependensi
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Konfigurasi Lingkungan (.env)

Salin berkas template lingkungan `.env.example` menjadi `.env`, lalu sesuaikan kredensial koneksi RTSP kamera:

```bash
copy .env.example .env
```

Sunting nilai variabel pada `.env`:
```ini
APP_ENV=production
LOG_LEVEL=INFO

# Kredensial Kamera 01 (Koridor Utama)
CAM_01_USER=admin
CAM_01_PASS=password_kamera_anda
CAM_01_HOST=192.168.1.101
CAM_01_PORT=554
CAM_01_PATH=/Streaming/Channels/102
```

---

### 3. Menjalankan Sistem Utama

#### Opsi A: Menjalankan via Skrip Windows (Direkomendasikan)
Skrip ini secara otomatis memvalidasi integritas environment, menjalankan audit skema JSON, meluncurkan browser, dan mengaktifkan engine:
```cmd
run_system.bat
```

#### Opsi B: Menjalankan Manual via CLI Python
```bash
python main.py --host 0.0.0.0 --port 8070
```

#### Opsi C: Menjalankan via Docker Compose
```bash
docker compose up -d --build
```
Setelah aplikasi berjalan, buka peramban web pada alamat:
`http://127.0.0.1:8070`

---

### 4. Kalibrasi ROI Spasial (GUI Calibrator)

Jika sudut kamera bergeser atau perlu mengubah batas petak parkir dan posisi garis tripwire, gunakan utilitas GUI Calibrator:

```bash
python tools/roi_calibrator.py --camera cam_01
```

- **Slot Parkir (Poligon)**: Klik 4 titik sudut petak secara berurutan, lalu beri nama `zone_01` s.d. `zone_08`.
- **Tripwire (Garis Gerbang)**: Tarik garis 2 titik (P1 ke P2) melintasi bibir petak, tentukan arah $A \rightarrow B$, dan pasangkan ID dengan format `tw_01` s.d. `tw_08`.
- Tekan **Save** untuk menyimpan konfigurasi langsung ke `cameras/cam_01/roi_zones.json`.

---

### 5. Verifikasi Pengujian & Regression Tests

Repositori ini dilengkapi suite pengujian otomatis untuk memvalidasi kestabilan state machine dan kekebalan sistem terhadap ID churn:

```bash
# 1. Uji Validasi Triple-Check & Anti-ID Churn (8 Skenario Komprehensif)
python tools/test_triple_check_parking.py

# 2. Uji Cold-Start Baseline Scan
python tools/test_cold_start_parking.py

# 3. Uji Integrasi Smart Parking Dasar
python tools/test_smart_parking.py
```

Seluruh pengujian dirancang untuk menghasilkan status `100% PASS` guna menjamin keandalan sistem sebelum proses deployment ke lini produksi.
