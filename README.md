# HexRingCalibrationTarget genarator, detector and calibrator

## Configure and build
```
cmake -S. -B build
cmake --build build
```

## Set output path
Default is `$HOME/.hextarget/DrawBoard` but can be set with:
```
export HEXTARGET_SAVE_PATH=$HOME/.hex
```

## Run drawing boards
### RectGridCalibTarget
```
./build/bin/calibration DrawBoard -r 3 --board 2 --dpi 1200 --board-params-path ./configs/board-rect.json
```
### HexGridCalibTarget
```
./build/bin/calibration DrawBoard -r 3 --board 2 --dpi 1200 --board-params-path ./configs/board-hex.json
```
### CircleGridCalibTarget
```
./build/bin/calibration DrawBoard -r 3 --board 2 --dpi 1200 --board-params-path ./configs/board-circle.json
```
