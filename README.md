# ベイズ推定におけるニューロンの時間スケールと役割分担の関係
本研究ではRecurrent Neural Networkを用いて、ベイズ推定を行うときのニューロンが持つ時間スケールとネットワーク内の情報処理の関係を調べた。分かったことは以下の通り
- priorがゆっくりと変化する環境において時間スケールが固定されている場合、遅いニューロンがpriorのコーディングに寄与し、元の値を推定する際のpriorを考慮した計算を担う
    - 上記の役割分担はニューロンの時間スケールが一定の場合は起こらない
    - 役割分担の結果、遅いニューロン側がpriorに関する制御を行う仕組みはモデルで理解できる
- 遅いニューロンと速いニューロンがあるニューラルネットワークの方がニューロンの時間スケールが一定のニューラルネットワークよりもより最適なベイズ推定が可能になっている
    - これは遅いニューロンが時間積分する際のtime windowを長く取ることができ、priorのコーディングに有利であると考えられる
    - 実際priorのコーディングの様子を調べると時間スケールが分かれているモデルの方が高精度である
    - 逆に全てのニューロンを遅くしてしまうとlikelihoodの速い変化にうまく対応できないため、遅いニューロンと速いニューロンが存在することが重要である
    - またここから役割分担することが推論にとって有利に働くことが分かる。
    - ベイズ推定の精度の違いは特にpriorの分散の影響を考える際に顕著になり、速いpriorの変化に対する対応において大きな差がある
- 時間スケールが学習可能な場合、一定の時間スケールから学習を始めた場合でも速いニューロンと遅いニューロンに分化する。
    - このときコードする情報の役割も分化し、構造にも変化が現れる
    - 遅いニューロンの時間スケールは環境の時間スケールに応じて変化する

## 学習
### 1. configの設定
以下のように学習時の設定を`config`ファイルに記述する。

```yaml
DATALOADER:
  TIME_LENGTH: 120
  INPUT_NEURON: 100
  UNCERTAINTY: 0.5
  SIGMA_SQ: 0.5
  G_MIN: 0.25
  G_MAX: 1.25
MACHINE:
  CUDA: true
  SEED: 1
MODEL:
  SIGMA_NEU: 0.1
  SIZE: 200
  RESERVOIR: 50
  ALPHA_FAST: 1
  ALPHA_SLOW: 0.1
TRAIN:
  BATCHSIZE: 50
  LR: 0.001
  NUM_EPOCH: 500
  NUM_SAVE_EPOCH: 10
  DISPLAY_EPOCH: 1
  OPT: Adam
  WEIGHT_DECAY: 0.0001
```

### 2. 学習の実行
```bash
$ python train_random_dynamic_state.py {cfg_path}
```

学習された重みは`slow-reservoir/slow_reservoir/trained_model/`以下に保存される。

## 解析
### 時間スケールが固定されたモデル
#### 役割分担の定量化
```bash
$ python division_of_roles.py \
    ../cfg/dynamic_state/20220428_3.cfg \
    ../trained_model/dynamic_state_random/20220428_3/epoch_500.pth \
    -sn 500
$ python division_of_roles.py {cfg_path} {model_path} -sn {sample_num}
```

結果は以下のように表示される。

```
Variation sub: 0.01090
Variation main: 0.00439
```

`Variation sub`はsub networkの状態を別のpriorによって到達する内部状態に変化させた時の出力の分散、
`Variation main`はmain networkの状態を別のpriorによって到達する内部状態に変化させた時の出力の分散を表す。

#### ベイズ推定の最適性
```bash
$ python bayesian_optimality.py \
     ../cfg/dynamic_state/20220526_all_slow.cfg \
     ../trained_model/dynamic_state_random/20220526_all_slow/epoch_500.pth \
     -tp 0.03 \
     -sn 500
$ python bayesian_optimality.py {cfg_path} {model_path} \
     -tp {transition_probability} -sn {sample_num}
```

結果は`Mean Squared Error`として出力される。
```
Mean Squared Error: 0.02802
```

#### priorを符号化する際の計算モデル

### 時間スケールが学習可能なモデル
#### 構造の解析

#### 遅いニューロンの時間スケール