# YOLOv5n Pure C Implementation

순수 C로 구현한 YOLOv5n(nano) 객체 탐지 추론 엔진. 외부 라이브러리 없이 동작하며, 호스트 빌드와 **Bare-metal(FPGA)** 빌드를 하나의 코드베이스로 지원한다.

## 목표

- **최종**: MicroBlaze V(RISC-V) 등 FPGA에서 YOLOv5n 추론 실행
- **제약**: OpenCV/OpenBLAS 등 미사용, 순수 C만 사용
- **상태**: Python YOLOv5n과 동일한 추론 결과 (호스트·보드 검증 완료)

## 폴더 구조

```
YOLOv5n_in_C/
├── assets/                     # 모델 파일
│   ├── yolov5n.pt              # PyTorch 원본 모델
│   └── weights.bin             # C용 변환된 가중치 (Fused)
│
├── csrc/                        # C 소스 코드
│   ├── main.c                  # 메인 추론 파이프라인
│   ├── platform_config.h       # BARE_METAL DDR 맵 / 매크로
│   │
│   ├── blocks/                  # 고수준 블록
│   │   ├── conv.c/h            # Conv 블록 (Conv2D + Bias + SiLU)
│   │   ├── c3.c/h              # C3 블록 (cv1 + cv2 + Bottleneck + cv3)
│   │   ├── sppf.c/h            # SPPF 블록 (Spatial Pyramid Pooling Fast)
│   │   ├── detect.c/h          # Detect Head (1×1 Conv × 3 스케일)
│   │   ├── decode.c/h          # Anchor-based Decode + hw_detection_t 정의
│   │   └── nms.c/h             # Non-Maximum Suppression
│   │
│   ├── operations/              # 저수준 연산
│   │   ├── conv2d.c/h          # 2D Convolution
│   │   ├── silu.c/h            # SiLU 활성화 함수
│   │   ├── bottleneck.c/h      # Bottleneck 모듈
│   │   ├── concat.c/h          # 채널 방향 Concat
│   │   ├── maxpool2d.c/h       # 2D Max Pooling
│   │   └── upsample.c/h        # Nearest Neighbor 2× Upsampling
│   │
│   └── utils/                   # 유틸리티
│       ├── weights_loader.c/h  # weights.bin 로더 (DDR 제로카피 지원)
│       ├── image_loader.c/h    # 전처리된 이미지 로더 (DDR 제로카피 지원)
│       ├── feature_pool.c/h    # 피처맵 풀 할당자 (버퍼 재사용)
│       └── uart_dump.c/h       # UART 검출 결과 덤프 (BARE_METAL)
│
├── data/
│   ├── image/                   # 입력 이미지
│   │   └── zidane.jpg
│   ├── input/                   # 전처리된 입력
│   │   └── preprocessed_image.bin
│   └── output/                  # 추론 결과
│       ├── detections.bin      # C 결과 (HW 바이너리 포맷)
│       ├── detections.txt      # C 결과 (텍스트)
│       ├── detections.jpg      # C 결과 시각화
│       └── ref/                 # Python 참조 결과
│           ├── detections.bin
│           ├── detections.txt
│           └── detections.jpg
│
├── tools/                        # Python 도구
│   ├── export_weights_to_bin.py # PyTorch → weights.bin 변환
│   ├── preprocess_image_to_bin.py # 이미지 전처리
│   ├── run_python_yolov5n_fused.py # Python 참조 출력 생성
│   ├── decode_detections.py     # bin → txt 변환 + 시각화
│   ├── recv_detections_uart.py  # UART 수신 → detections.bin (BARE_METAL용)
│   ├── uart_to_detections_txt.py # UART 수신 → detections.txt(.jpg) 한 번에
│   ├── verify_weights_bin.py    # weights.bin 형식 검증
│   ├── reweight_align4.py       # weights.bin 4바이트 정렬 패딩 추가
│   └── gen_test_vectors.py      # 테스트 벡터 생성
│
├── tests/                        # 단위 테스트
│   ├── test_*.c                 # 각 블록별 테스트
│   └── test_vectors_*.h         # 테스트 벡터
│
├── CHANGELOG.md                  # 변경 이력
├── VITIS_BUILD.md               # Vitis Bare-metal 빌드·메모리 맵·링커 스크립트
└── TESTING.md                   # 테스트 방법
```

## 빌드 및 실행

### 호스트 (Linux/macOS)

```bash
gcc -o main csrc/main.c csrc/blocks/*.c csrc/operations/*.c csrc/utils/*.c \
    -I. -Icsrc -lm -std=c99 -O2
./main
```

입력: `data/input/preprocessed_image.bin`, 가중치: `assets/weights.bin` (파일에서 로드).  
출력: `data/output/detections.bin` (1바이트 개수 + 12바이트×N 검출).

### Bare-metal (Vitis, Arty A7 + MicroBlaze V 등)

- 컴파일 옵션: `-DBARE_METAL`, include: `csrc`
- 입력/가중치: DDR 고정 주소에서 직접 참조 (파일 I/O 없음)
- 출력: DDR `DETECTIONS_OUT_BASE` 버퍼 + UART Hex 덤프

상세 메모리 맵, 캐시, 링커 스크립트, UART 프로토콜은 **[VITIS_BUILD.md](VITIS_BUILD.md)** 참고.

## 워크플로우 요약

1. **가중치**: `tools/export_weights_to_bin.py` → `assets/weights.bin`
2. **이미지 전처리**: `tools/preprocess_image_to_bin.py` → `data/input/preprocessed_image.bin`
3. **C 추론**: `./main` (호스트) 또는 보드에서 실행
4. **결과 확인**: `tools/decode_detections.py` 로 bin → txt/시각화

Bare-metal 보드에서는 가중치·이미지를 DDR에 미리 적재한 뒤 실행하며, 결과는 UART로 받아 `tools/recv_detections_uart.py` 등으로 저장 후 동일하게 디코딩 가능.

## 기술 요약

- **Fused 모델**: Conv+BN → Conv+Bias로 흡수, BN 연산 제거
- **NCHW**: 모든 텐서가 Batch×Channel×Height×Width
- **Anchor-based**: P3/P4/P5 각 3앵커, 255ch = 3×85 (bbox+obj+80클래스)
- **HW 출력**: 12바이트/검출 (x,y,w,h, class_id, confidence 등), 상세는 `decode.h` 의 `hw_detection_t`

## 테스트

블록별 단위 테스트 및 벡터 생성 방법은 **[TESTING.md](TESTING.md)** 참고.

## 라이선스 / 참고

- YOLOv5 계열 모델·가중치 사용 시 Ultralytics 라이선스 확인
- 변경 이력: [CHANGELOG.md](CHANGELOG.md)
